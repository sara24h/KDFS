import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, meter, scheduler
from data.dataset import Dataset_hardfakevsreal  # اضافه کردن دیتاست جدید
import torchvision.models as models

class Train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.resume = args.resume

        self.start_epoch = 0
        self.best_prec1 = 0

    def result_init(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
        self.logger = utils.get_logger(
            os.path.join(self.result_dir, "train_logger.log"), "train_logger"
        )
        self.logger.info("train config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(
            self.args, os.path.join(self.result_dir, "train_config.txt")
        )
        self.logger.info("--------- Train -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        self.train_loader, self.val_loader = Dataset_hardfakevsreal.get_loaders(
            self.dataset_dir,
            os.path.join(self.dataset_dir, 'data.csv'),
            self.train_batch_size,
            self.eval_batch_size,
            self.num_workers,
            self.pin_memory,
            ddp=False
        )
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")
        self.logger.info("Loading ResNet50 model")
        self.model = models.resnet50(pretrained=True)  # مدل از پیش آموزش‌دیده ImageNet

        # تغییر لایه نهایی برای 2 کلاس
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # فریز کردن لایه‌های اولیه
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.logger.info("Model has been built!")

    def define_loss(self):
        # وزن‌دهی به کلاس‌ها برای مدیریت نامتوازنی (Fake: 0, Real: 1)
        class_weights = torch.tensor([589/1288, 700/1288]).to(self.device)
        self.ori_loss = nn.CrossEntropyLoss(weight=class_weights)

    def define_optim(self):
        self.optim_weight = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr
        )

    def resume_student_ckpt(self):
        ckpt = torch.load(self.resume)
        self.best_prec1 = ckpt["best_prec1"]
        self.start_epoch = ckpt["start_epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.optim_weight.load_state_dict(ckpt["optim_weight"])
        self.scheduler_student_weight.load_state_dict(
            ckpt["scheduler_student_weight"]
        )
        self.logger.info("=> Continue from epoch {}...".format(self.start_epoch))

    def save_student_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt = {
            "best_prec1": self.best_prec1,
            "start_epoch": self.start_epoch,
            "model": self.model.state_dict(),
            "optim_weight": self.optim_weight.state_dict(),
            "scheduler_student_weight": self.scheduler_student_weight.state_dict()
        }

        if is_best:
            torch.save(ckpt, os.path.join(folder, "resnet50_best.pt"))
        torch.save(ckpt, os.path.join(folder, "resnet50_last.pt"))

    def train(self):
        if self.device == "cuda":
            self.model = self.model.cuda()
            self.ori_loss = self.ori_loss.cuda()

        if self.resume:
            self.resume_student_ckpt()

        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            # train
            self.model.train()
            meter_oriloss.reset()
            meter_loss.reset()
            meter_top1.reset()
            meter_top5.reset()
            lr = self.optim_weight.state_dict()["param_groups"][0]["lr"]

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                for images, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits = self.model(images)  # ResNet50 مستقیماً logits را تولید می‌کند
                    ori_loss = self.ori_loss(logits, targets)
                    total_loss = ori_loss
                    total_loss.backward()
                    self.optim_weight.step()

                    prec1, prec5 = utils.get_accuracy(logits, targets, topk=(1, 5))
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)

                    _tqdm.set_postfix(loss="{:.4f}".format(meter_loss.avg), top1="{:.4f}".format(meter_top1.avg))
                    _tqdm.update(1)
                    time.sleep(0.01)

            self.scheduler_student_weight.step()
            self.writer.add_scalar("train/loss/ori_loss", meter_oriloss.avg, global_step=epoch)
            self.writer.add_scalar("train/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("train/acc/top5", meter_top5.avg, global_step=epoch)
            self.writer.add_scalar("train/lr/lr", lr, global_step=epoch)

            self.logger.info(
                "[Train] Epoch {0} : LR {lr:.6f} OriLoss {ori_loss:.4f} TotalLoss {total_loss:.4f} Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch, lr=lr, ori_loss=meter_oriloss.avg, total_loss=meter_loss.avg, top1=meter_top1.avg, top5=meter_top5.avg
                )
            )

            # valid
            self.model.eval()
            meter_top1.reset()
            meter_top5.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits = self.model(images)
                        prec1, prec5 = utils.get_accuracy(logits, targets, topk=(1, 5))
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        meter_top5.update(prec5.item(), n)
                        _tqdm.set_postfix(top1="{:.4f}".format(meter_top1.avg), top5="{:.4f}".format(meter_top5.avg))
                        _tqdm.update(1)
                        time.sleep(0.01)

            self.writer.add_scalar("val/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("val/acc/top5", meter_top5.avg, global_step=epoch)

            self.logger.info("[Val] Epoch {0} : Prec@(1,5) {top1:.2f}, {top5:.2f}".format(epoch, top1=meter_top1.avg, top5=meter_top5.avg))

            self.start_epoch += 1
            if self.best_prec1 < meter_top1.avg:
                self.best_prec1 = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)

            self.logger.info(" => Best top1 accuracy: " + str(self.best_prec1))
        self.logger.info("Training finished!")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
