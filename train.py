import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, meter, scheduler
from data.dataset import Dataset_hardfakevsreal
from torchvision.models import resnet50, ResNet50_Weights

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
        self.teacher_dir = args.teacher_dir
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

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
        self.logger.info(str(vars(self.args)))
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
        if self.dataset_type != 'hardfakevsrealfaces':
            raise ValueError("Only 'hardfakevsrealfaces' dataset is supported in this configuration")
        
        self.train_loader, self.val_loader, self.test_loader = Dataset_hardfakevsreal.get_loaders(
            self.dataset_dir,
            self.args.csv_file if hasattr(self.args, 'csv_file') else os.path.join(self.dataset_dir, 'train.csv'),
            self.train_batch_size,
            self.eval_batch_size,
            self.num_workers,
            self.pin_memory,
            self.args.ddp if hasattr(self.args, 'ddp') else False,
            None
        )
        self.logger.info("Dataset has been loaded! Train: {}, Val: {}, Test: {}".format(
            len(self.train_loader.dataset), len(self.val_loader.dataset), len(self.test_loader.dataset if self.test_loader else [])))

    def build_model(self):
        self.logger.info("==> Building teacher model..")
        self.teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.teacher.parameters():
            param.requires_grad = False
        num_ftrs = self.teacher.fc.in_features
        self.teacher.fc = nn.Linear(num_ftrs, 2)
        for param in self.teacher.fc.parameters():
            param.requires_grad = True
        self.teacher = self.teacher.to(self.device)
        self.logger.info("Teacher model built successfully!")

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss()

    def define_optim(self):
        self.teacher_optimizer = torch.optim.Adam(
            self.teacher.fc.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.teacher_scheduler = scheduler.CosineAnnealingLRWarmup(
            self.teacher_optimizer,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )

    def save_teacher_ckpt(self, is_best, epoch):
        folder = os.path.join(self.args.teacher_dir, "teacher_model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_teacher = {
            "epoch": epoch,
            "best_prec1": self.best_prec1,
            "state_dict": self.teacher.state_dict(),
        }

        if is_best:
            torch.save(
                ckpt_teacher,
                os.path.join(folder, "resnet50_teacher_best.pt"),
            )
        torch.save(
            ckpt_teacher,
            os.path.join(folder, "resnet50_teacher_last.pt"),
        )

    def train_teacher(self):
        if self.device == "cuda":
            self.teacher = self.teacher.cuda()
            self.ori_loss = self.ori_loss.cuda()

        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        for epoch in range(1, self.num_epochs + 1):
            self.teacher.train()
            meter_loss.reset()
            meter_top1.reset()
            meter_top5.reset()
            lr = self.teacher_optimizer.state_dict()["param_groups"][0]["lr"]

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description("teacher epoch: {}/{}".format(epoch, self.num_epochs))
                for images, targets in self.train_loader:
                    self.teacher_optimizer.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    
                    logits = self.teacher(images)
                    loss = self.ori_loss(logits, targets)
                    
                    loss.backward()
                    self.teacher_optimizer.step()

                    prec1, prec5 = utils.get_accuracy(logits, targets, topk=(1, 5))
                    n = images.size(0)
                    meter_loss.update(loss.item(), n)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)

                    _tqdm.set_postfix(
                        loss="{:.4f}".format(meter_loss.avg),
                        top1="{:.4f}".format(meter_top1.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

            self.teacher_scheduler.step()

            self.writer.add_scalar("teacher_train/loss", meter_loss.avg, global_step=epoch)
            self.writer.add_scalar("teacher_train/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("teacher_train/acc/top5", meter_top5.avg, global_step=epoch)
            self.writer.add_scalar("teacher_train/lr", lr, global_step=epoch)

            self.logger.info(
                "[Teacher Train] Epoch {0} : LR {lr:.6f} Loss {loss:.4f} Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch, lr=lr, loss=meter_loss.avg, top1=meter_top1.avg, top5=meter_top5.avg
                )
            )

            self.teacher.eval()
            meter_top1.reset()
            meter_top5.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description("teacher val epoch: {}/{}".format(epoch, self.num_epochs))
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits = self.teacher(images)
                        prec1, prec5 = utils.get_accuracy(logits, targets, topk=(1, 5))
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        meter_top5.update(prec5.item(), n)

                        _tqdm.set_postfix(
                            top1="{:.4f}".format(meter_top1.avg),
                            top5="{:.4f}".format(meter_top5.avg),
                        )
                        _tqdm.update(1)
                        time.sleep(0.01)

            self.writer.add_scalar("teacher_val/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("teacher_val/acc/top5", meter_top5.avg, global_step=epoch)

            self.logger.info(
                "[Teacher Val] Epoch {0} : Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch, top1=meter_top1.avg, top5=meter_top5.avg
                )
            )

            is_best = meter_top1.avg > self.best_prec1
            if is_best:
                self.best_prec1 = meter_top1.avg
            self.save_teacher_ckpt(is_best, epoch)

        self.logger.info("Teacher training finished!")

    def test_teacher(self):
        self.teacher.eval()
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                _tqdm.set_description("teacher test")
                for images, targets in self.val_loader:
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits = self.teacher(images)
                    prec1, prec5 = utils.get_accuracy(logits, targets, topk=(1, 5))
                    n = images.size(0)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)

                    _tqdm.set_postfix(
                        top1="{:.4f}".format(meter_top1.avg),
                        top5="{:.4f}".format(meter_top5.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

        self.writer.add_scalar("teacher_test/acc/top1", meter_top1.avg)
        self.writer.add_scalar("teacher_test/acc/top5", meter_top5.avg)

        self.logger.info(
            "[Teacher Test] Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                top1=meter_top1.avg, top5=meter_top5.avg
            )
        )

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train_teacher()
        self.test_teacher()
