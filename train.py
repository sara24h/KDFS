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
from utils import utils, loss, meter, scheduler
from data.dataset import Dataset_cifar10, Dataset_cifar100, Dataset_imagenet, Dataset_hardfakevsreal
from model.teacher.resnet_cifar import (
    resnet_56_cifar10,
    resnet_110_cifar10,
    resnet_56_cifar100,
)
from model.student.resnet_sparse_cifar import (
    resnet_56_sparse_cifar10,
    resnet_110_sparse_cifar10,
    resnet_56_sparse_cifar100,
)
from model.teacher.ResNet import ResNet_18_imagenet, ResNet_50_imagenet
from model.student.ResNet_sparse import (
    ResNet_18_sparse_imagenet,
    ResNet_50_sparse_imagenet,
)
from model.teacher.vgg_bn import VGG_16_bn_cifar10
from model.student.vgg_bn_sparse import VGG_16_bn_sparse_cifar10
from model.teacher.DenseNet import DenseNet_40_cifar10
from model.student.DenseNet_sparse import (
    DenseNet_40_sparse_cifar10,
)
from model.teacher.GoogLeNet import GoogLeNet_cifar10
from model.student.GoogLeNet_sparse import GoogLeNet_sparse_cifar10
from model.teacher.MobileNetV2 import MobileNetV2_imagenet
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_imagenet
from torchvision.models import resnet50, ResNet50_Weights

Flops_baselines = {
    "resnet_56": 125.49,
    "resnet_110": 252.89,
    "ResNet_18": 1820,
    "ResNet_50": 4134,
    "VGG_16_bn": 313.73,
    "DenseNet_40": 282.00,
    "GoogLeNet": 1520,
    "MobileNetV2": 327.55,
}

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
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        self.compress_rate = args.compress_rate
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
        dataset_classes = {
            'cifar10': Dataset_cifar10,
            'cifar100': Dataset_cifar100,
            'imagenet': Dataset_imagenet,
            'hardfakevsrealfaces': Dataset_hardfakevsreal
        }
        dataset = dataset_classes[self.dataset_type](
            self.dataset_dir,
            self.args.csv_file if hasattr(self.args, 'csv_file') else os.path.join(self.dataset_dir, 'train.csv'),
            self.train_batch_size,
            self.eval_batch_size,
            self.num_workers,
            self.pin_memory,
            self.args.ddp if hasattr(self.args, 'ddp') else False,
            None  # test_csv_file نداریم
        )
        self.train_loader, self.val_loader, self.test_loader = dataset.get_loaders(
            self.dataset_dir,
            self.args.csv_file if hasattr(self.args, 'csv_file') else os.path.join(self.dataset_dir, 'train.csv'),
            self.train_batch_size,
            self.eval_batch_size,
            self.num_workers,
            self.pin_memory,
            self.args.ddp if hasattr(self.args, 'ddp') else False,
            None
        )
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building models..")

        self.logger.info("Loading teacher model")
        if self.dataset_type == 'hardfakevsrealfaces':
            self.teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            for param in self.teacher.parameters():
                param.requires_grad = False
            num_ftrs = self.teacher.fc.in_features
            self.teacher.fc = nn.Linear(num_ftrs, 2)
            for param in self.teacher.fc.parameters():
                param.requires_grad = True
        else:
            self.teacher = eval(self.arch + "_" + self.dataset_type)()
            ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
            if self.arch in ["resnet_56", "resnet_110", "VGG_16_bn", "DenseNet_40", "GoogLeNet"]:
                self.teacher.load_state_dict(ckpt_teacher["state_dict"])
            elif self.arch in ["ResNet_18", "ResNet_50", "MobileNetV2"]:
                self.teacher.load_state_dict(ckpt_teacher)

        self.logger.info("Building student model")
        self.student = eval(self.arch + "_sparse_" + self.dataset_type)(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss()
        self.kd_loss = loss.KDLoss()
        self.rc_loss = loss.RCLoss()
        self.mask_loss = loss.MaskLoss()

    def define_optim(self):
        if self.dataset_type == 'hardfakevsrealfaces':
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
        else:
            weight_params = map(
                lambda a: a[1],
                filter(
                    lambda p: p[1].requires_grad and "mask" not in p[0],
                    self.student.named_parameters(),
                ),
            )
            mask_params = map(
                lambda a: a[1],
                filter(
                    lambda p: p[1].requires_grad and "mask" in p[0],
                    self.student.named_parameters(),
                ),
            )
            self.optim_weight = torch.optim.Adamax(
                weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7
            )
            self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)
            self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
                self.optim_weight,
                T_max=self.lr_decay_T_max,
                eta_min=self.lr_decay_eta_min,
                last_epoch=-1,
                warmup_steps=self.warmup_steps,
                warmup_start_lr=self.warmup_start_lr,
            )
            self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
                self.optim_mask,
                T_max=self.lr_decay_T_max,
                eta_min=self.lr_decay_eta_min,
                last_epoch=-1,
                warmup_steps=self.warmup_steps,
                warmup_start_lr=self.warmup_start_lr,
            )

    def resume_student_ckpt(self):
        ckpt_student = torch.load(self.resume)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.scheduler_student_weight.load_state_dict(
            ckpt_student["scheduler_student_weight"]
        )
        self.scheduler_student_mask.load_state_dict(
            ckpt_student["scheduler_student_mask"]
        )
        self.logger.info("=> Continue from epoch {}...".format(self.start_epoch))

    def save_student_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_student = {}
        ckpt_student["best_prec1"] = self.best_prec1
        ckpt_student["start_epoch"] = self.start_epoch
        ckpt_student["student"] = self.student.state_dict()
        ckpt_student["optim_weight"] = self.optim_weight.state_dict()
        ckpt_student["optim_mask"] = self.optim_mask.state_dict()
        ckpt_student[
            "scheduler_student_weight"
        ] = self.scheduler_student_weight.state_dict()
        ckpt_student[
            "scheduler_student_mask"
        ] = self.scheduler_student_mask.state_dict()

        if is_best:
            torch.save(
                ckpt_student,
                os.path.join(folder, self.arch + "_sparse_best.pt"),
            )
        torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))

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

    def train(self):
        if self.device == "cuda":
            self.teacher = self.teacher.cuda()
            self.student = self.student.cuda()
            self.ori_loss = self.ori_loss.cuda()
            self.kd_loss = self.kd_loss.cuda()
            self.rc_loss = self.rc_loss.cuda()
            self.mask_loss = self.mask_loss.cuda()

        if self.resume:
            self.resume_student_ckpt()

        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
        meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
        meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        self.teacher.eval()
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.student.train()
            self.student.ticket = False
            meter_oriloss.reset()
            meter_kdloss.reset()
            meter_rcloss.reset()
            meter_maskloss.reset()
            meter_loss.reset()
            meter_top1.reset()
            meter_top5.reset()
            lr = (
                self.optim_weight.state_dict()["param_groups"][0]["lr"]
                if epoch > 1
                else self.warmup_start_lr
            )

            self.student.update_gumbel_temperature(epoch)
            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                for images, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits_student, feature_list_student = self.student(images)
                    with torch.no_grad():
                        logits_teacher, feature_list_teacher = self.teacher(images)
                    ori_loss = self.ori_loss(logits_student, targets)
                    kd_loss = (self.target_temperature**2) * self.kd_loss(
                        logits_teacher / self.target_temperature,
                        logits_student / self.target_temperature,
                    )
                    rc_loss = torch.tensor(0)
                    for i in range(len(feature_list_student)):
                        rc_loss = rc_loss + self.rc_loss(
                            feature_list_student[i], feature_list_teacher[i]
                        )
                    Flops_baseline = Flops_baselines[self.arch]
                    Flops = self.student.get_flops()
                    mask_loss = self.mask_loss(
                        Flops, Flops_baseline * (10**6), self.compress_rate
                    )
                    total_loss = (
                        ori_loss
                        + self.coef_kdloss * kd_loss
                        + self.coef_rcloss * rc_loss / len(feature_list_student)
                        + self.coef_maskloss * mask_loss
                    )
                    total_loss.backward()
                    self.optim_weight.step()
                    self.optim_mask.step()
                    prec1, prec5 = utils.get_accuracy(
                        logits_student, targets, topk=(1, 5)
                    )
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_kdloss.update(self.coef_kdloss * kd_loss.item(), n)
                    meter_rcloss.update(
                        self.coef_rcloss * rc_loss.item() / len(feature_list_student), n
                    )
                    meter_maskloss.update(self.coef_maskloss * mask_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)
                    _tqdm.set_postfix(
                        loss="{:.4f}".format(meter_loss.avg),
                        top1="{:.4f}".format(meter_top1.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)
            Flops = self.student.get_flops()
            self.scheduler_student_weight.step()
            self.scheduler_student_mask.step()
            self.writer.add_scalar("train/loss/ori_loss", meter_oriloss.avg, global_step=epoch)
            self.writer.add_scalar("train/loss/kd_loss", meter_kdloss.avg, global_step=epoch)
            self.writer.add_scalar("train/loss/rc_loss", meter_rcloss.avg, global_step=epoch)
            self.writer.add_scalar("train/loss/mask_loss", meter_maskloss.avg, global_step=epoch)
            self.writer.add_scalar("train/loss/total_loss", meter_loss.avg, global_step=epoch)
            self.writer.add_scalar("train/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("train/acc/top5", meter_top5.avg, global_step=epoch)
            self.writer.add_scalar("train/lr/lr", lr, global_step=epoch)
            self.writer.add_scalar(
                "train/temperature/gumbel_temperature",
                self.student.gumbel_temperature,
                global_step=epoch,
            )
            self.writer.add_scalar("train/Flops", Flops, global_step=epoch)
            self.logger.info(
                "[Train] "
                "Epoch {0} : "
                "Gumbel_temperature {gumbel_temperature:.2f} "
                "LR {lr:.6f} "
                "OriLoss {ori_loss:.4f} "
                "KDLoss {kd_loss:.4f} "
                "RCLoss {rc_loss:.4f} "
                "MaskLoss {mask_loss:.6f} "
                "TotalLoss {total_loss:.4f} "
                "Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch,
                    gumbel_temperature=self.student.gumbel_temperature,
                    lr=lr,
                    ori_loss=meter_oriloss.avg,
                    kd_loss=meter_kdloss.avg,
                    rc_loss=meter_rcloss.avg,
                    mask_loss=meter_maskloss.avg,
                    total_loss=meter_loss.avg,
                    top1=meter_top1.avg,
                    top5=meter_top5.avg,
                )
            )
            masks = []
            for _, m in enumerate(self.student.mask_modules):
                masks.append(round(m.mask.mean().item(), 2))
            self.logger.info("[Train mask avg] Epoch {0} : ".format(epoch) + str(masks))
            self.logger.info(
                "[Train model Flops] Epoch {0} : ".format(epoch)
                + str(Flops.item() / (10**6))
                + "M"
            )
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            meter_top5.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits_student, _ = self.student(images)
                        prec1, prec5 = utils.get_accuracy(
                            logits_student, targets, topk=(1, 5)
                        )
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        meter_top5.update(prec5.item(), n)
                        _tqdm.set_postfix(
                            top1="{:.4f}".format(meter_top1.avg),
                            top5="{:.4f}".format(meter_top5.avg),
                        )
                        _tqdm.update(1)
                        time.sleep(0.01)
            Flops = self.student.get_flops()
            self.writer.add_scalar("val/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("val/acc/top5", meter_top5.avg, global_step=epoch)
            self.writer.add_scalar("val/Flops", Flops, global_step=epoch)
            self.logger.info(
                "[Val] "
                "Epoch {0} : "
                "Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch,
                    top1=meter_top1.avg,
                    top5=meter_top5.avg,
                )
            )
            masks = []
            for _, m in enumerate(self.student.mask_modules):
                masks.append(round(m.mask.mean().item(), 2))
            self.logger.info("[Val mask avg] Epoch {0} : ".format(epoch) + str(masks))
            self.logger.info(
                "[Val model Flops] Epoch {0} : ".format(epoch)
                + str(Flops.item() / (10**6))
                + "M"
            )
            self.start_epoch += 1
            if self.best_prec1 < meter_top1.avg:
                self.best_prec1 = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)
            self.logger.info(
                " => Best top1 accuracy before finetune : " + str(self.best_prec1)
            )
        self.logger.info("Training finished!")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        if self.dataset_type == 'hardfakevsrealfaces':
            self.train_teacher()
            self.test_teacher()
        else:
            self.train()
