import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import models
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from utils import utils, loss, meter, scheduler
from thop import profile

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

Flops_baselines = {
    "ResNet_50": {
        "hardfakevsrealfaces": 7690.0,
        "rvf10k": 5000.0,
    }
}

class Train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
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

        if self.dataset_mode == "hardfake":
            self.args.dataset_type = "hardfakevsrealfaces"
            self.num_classes = 1  # تغییر به 1 برای خروجی باینری
            self.image_size = 300
        elif self.dataset_mode == "rvf10k":
            self.args.dataset_type = "rvf10k"
            self.num_classes = 1  # تغییر به 1 برای خروجی باینری
            self.image_size = 256
        else:
            raise ValueError("dataset_mode must be 'hardfake' or 'rvf10k'")

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
        if self.dataset_mode not in ['hardfake', 'rvf10k']:
            raise ValueError("dataset_mode must be 'hardfake' or 'rvf10k'")

        if self.dataset_mode == 'hardfake':
            hardfake_csv_file = os.path.join(self.dataset_dir, 'data.csv')
            hardfake_root_dir = self.dataset_dir
            rvf10k_train_csv = None
            rvf10k_valid_csv = None
            rvf10k_root_dir = None
        else:
            hardfake_csv_file = None
            hardfake_root_dir = None
            rvf10k_train_csv = os.path.join(self.dataset_dir, 'train.csv')
            rvf10k_valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
            rvf10k_root_dir = self.dataset_dir

        if self.dataset_mode == 'hardfake' and not os.path.exists(hardfake_csv_file):
            raise FileNotFoundError(f"CSV file not found: {hardfake_csv_file}")
        if self.dataset_mode == 'rvf10k':
            if not os.path.exists(rvf10k_train_csv):
                raise FileNotFoundError(f"Train CSV file not found: {rvf10k_train_csv}")
            if not os.path.exists(rvf10k_valid_csv):
                raise FileNotFoundError(f"Valid CSV file not found: {rvf10k_valid_csv}")

        dataset_instance = Dataset_selector(
            dataset_mode=self.dataset_mode,
            hardfake_csv_file=hardfake_csv_file,
            hardfake_root_dir=hardfake_root_dir,
            rvf10k_train_csv=rvf10k_train_csv,
            rvf10k_valid_csv=rvf10k_valid_csv,
            rvf10k_root_dir=rvf10k_root_dir,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=False
        )

        self.train_loader = dataset_instance.loader_train
        self.val_loader = dataset_instance.loader_test
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")
        self.logger.info("Loading teacher model")
        # تغییر مدل معلم به resnet50 با یک خروجی
        self.teacher = models.resnet50(pretrained=False)
        num_ftrs = self.teacher.fc.in_features
        self.teacher.fc = nn.Linear(num_ftrs, 1)  # خروجی باینری
        ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu", weights_only=True)
        self.teacher.load_state_dict(ckpt_teacher, strict=True)

        self.logger.info("Building student model")
        if self.dataset_mode == "hardfake":
            self.student = ResNet_50_sparse_hardfakevsreal(
                gumbel_start_temperature=self.gumbel_start_temperature,
                gumbel_end_temperature=self.gumbel_end_temperature,
                num_epochs=self.num_epochs,
            )
        else:
            self.student = ResNet_50_sparse_rvf10k(
                gumbel_start_temperature=self.gumbel_start_temperature,
                gumbel_end_temperature=self.gumbel_end_temperature,
                num_epochs=self.num_epochs,
            )
        self.student.dataset_type = self.args.dataset_type

        # تغییر لایه نهایی دانش‌آموز به یک خروجی
        num_ftrs = self.student.fc.in_features
        self.student.fc = nn.Linear(num_ftrs, 1)

    def define_loss(self):
        # تغییر معیارها به BCEWithLogitsLoss
        self.ori_loss = nn.BCEWithLogitsLoss()
        self.kd_loss = nn.BCEWithLogitsLoss()  # تغییر به BCE برای تقطیر دانش
        self.rc_loss = loss.RCLoss()
        self.mask_loss = loss.MaskLoss()

    def define_optim(self):
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
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
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
        self.logger.info("=> Continue from epoch {}...".format(self.start_epoch + 1))

    def save_student_ckpt(self, is_best, epoch):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_student = {}
        ckpt_student["best_prec1"] = self.best_prec1
        ckpt_student["start_epoch"] = epoch
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

    def train(self):
        self.logger.info(f"Starting training from epoch: {self.start_epoch + 1}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
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
                        targets = targets.cuda().float()  # تبدیل به float

                    logits_student, feature_list_student = self.student(images)
                    logits_student = logits_student.squeeze(1)  # تبدیل به تک‌بعدی
                    with torch.no_grad():
                        logits_teacher, feature_list_teacher = self.teacher(images)
                        logits_teacher = logits_teacher.squeeze(1)  # تبدیل به تک‌بعدی

                    ori_loss = self.ori_loss(logits_student, targets)
                    kd_loss = self.kd_loss(logits_teacher, logits_student)

                    rc_loss = torch.tensor(0, device=images.device)
                    for i in range(len(feature_list_student)):
                        rc_loss = rc_loss + self.rc_loss(
                            feature_list_student[i], feature_list_teacher[i]
                        )

                    Flops_baseline = Flops_baselines[self.arch][self.args.dataset_type]
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

                    # محاسبه دقت برای خروجی باینری
                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100. * correct / images.size(0)
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_kdloss.update(self.coef_kdloss * kd_loss.item(), n)
                    meter_rcloss.update(
                        self.coef_rcloss * rc_loss.item() / len(feature_list_student), n
                    )
                    meter_maskloss.update(self.coef_maskloss * mask_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1, n)

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
            self.writer.add_scalar("train/lr/lr", lr, global_step=epoch)
            self.writer.add_scalar("train/temperature/gumbel_temperature", self.student.gumbel_temperature, global_step=epoch)
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
                "Prec@(1,) {top1:.2f}".format(
                    epoch,
                    gumbel_temperature=self.student.gumbel_temperature,
                    lr=lr,
                    ori_loss=meter_oriloss.avg,
                    kd_loss=meter_kdloss.avg,
                    rc_loss=meter_rcloss.avg,
                    mask_loss=meter_maskloss.avg,
                    total_loss=meter_loss.avg,
                    top1=meter_top1.avg,
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

            # Validation
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()

            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda().float()
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze(1)

                        # محاسبه دقت برای اعتبارسنجی
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100. * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        _tqdm.set_postfix(top1="{:.4f}".format(meter_top1.avg))
                        _tqdm.update(1)
                        time.sleep(0.01)

            Flops = self.student.get_flops()
            self.writer.add_scalar("val/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("val/Flops", Flops, global_step=epoch)

            self.logger.info(
                "[Val] "
                "Epoch {0} : "
                "Prec@(1,) {top1:.2f}".format(
                    epoch,
                    top1=meter_top1.avg,
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

            if self.best_prec1 < meter_top1.avg:
                self.best_prec1 = meter_top1.avg
                self.save_student_ckpt(True, epoch)
            else:
                self.save_student_ckpt(False, epoch)

            self.logger.info(
                " => Best top1 accuracy before finetune : " + str(self.best_prec1)
            )
        self.logger.info("Train finished!")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
