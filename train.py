import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from data.dataset import Dataset_hardfakevsreal  # دیتاست ارائه‌شده
# فرض می‌کنیم ماژول‌های utils، loss، meter، و scheduler در دسترس هستند
from utils import utils, loss, meter, scheduler

Flops_baselines = {
    "ResNet_50": 4134,
}

class Train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.csv_file = args.csv_file  # فایل CSV برای دیتاست
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
        self.train_loader, self.val_loader = Dataset_hardfakevsreal.get_loaders(
            data_dir=self.dataset_dir,
            csv_file=self.csv_file,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=False
        )
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")

        self.logger.info("Loading teacher model")
        self.teacher = models.resnet50()
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, 2)  # تنظیم برای 2 کلاس
        ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
        self.teacher.load_state_dict(ckpt_teacher)

        self.logger.info("Building student model")
        self.student = ResNet_50_sparse_imagenet(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )

    def define_loss(self):
        # مدیریت عدم تعادل کلاس‌ها
        class_weights = torch.tensor([1.0, 1.0]).to(self.device)  # قابل تنظیم بر اساس توزیع
        self.ori_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.kd_loss = loss.KDLoss()
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
        self.logger.info(f"=> Continue from epoch {self.start_epoch}...")

    def save_student_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)
        ckpt_student = {
            "best_prec1": self.best_prec1,
            "start_epoch": self.start_epoch,
            "student": self.student.state_dict(),
            "optim_weight": self.optim_weight.state_dict(),
            "optim_mask": self.optim_mask.state_dict(),
            "scheduler_student_weight": self.scheduler_student_weight.state_dict(),
            "scheduler_student_mask": self.scheduler_student_mask.state_dict(),
        }
        if is_best:
            torch.save(
                ckpt_student,
                os.path.join(folder, f"{self.arch}_sparse_best.pt"),
            )
        torch.save(ckpt_student, os.path.join(folder, f"{self.arch}_sparse_last.pt"))

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

        # Early Stopping
        patience = 5
        counter = 0
        best_val_loss = float('inf')

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
                _tqdm.set_description(f"epoch: {epoch}/{self.num_epochs}")
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
                    rc_loss = torch.tensor(0.0, device=logits_student.device)
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

                    prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_kdloss.update(self.coef_kdloss * kd_loss.item(), n)
                    meter_rcloss.update(
                        self.coef_rcloss * rc_loss.item() / len(feature_list_student), n
                    )
                    meter_maskloss.update(self.coef_maskloss * mask_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)

                    _tqdm.set_postfix(
                        loss="{:.4f}".format(meter_loss.avg),
                        top1="{:.4f}".format(meter_top1.avg),
                    )
                    _tqdm.update(1)

            Flops = self.student.get_flops()
            self.scheduler_student_weight.step()
            self.scheduler_student_mask.step()

            self.writer.add_scalar("train/loss/ori_loss", meter_oriloss.avg, epoch)
            self.writer.add_scalar("train/loss/kd_loss", meter_kdloss.avg, epoch)
            self.writer.add_scalar("train/loss/rc_loss", meter_rcloss.avg, epoch)
            self.writer.add_scalar("train/loss/mask_loss", meter_maskloss.avg, epoch)
            self.writer.add_scalar("train/loss/total_loss", meter_loss.avg, epoch)
            self.writer.add_scalar("train/acc/top1", meter_top1.avg, epoch)
            self.writer.add_scalar("train/lr/lr", lr, epoch)
            self.writer.add_scalar(
                "train/temperature/gumbel_temperature",
                self.student.gumbel_temperature,
                epoch,
            )
            self.writer.add_scalar("train/Flops", Flops, epoch)

            self.logger.info(
                f"[Train] Epoch {epoch} : "
                f"Gumbel_temperature {self.student.gumbel_temperature:.2f} "
                f"LR {lr:.6f} "
                f"OriLoss {meter_oriloss.avg:.4f} "
                f"KDLoss {meter_kdloss.avg:.4f} "
                f"RCLoss {meter_rcloss.avg:.4f} "
                f"MaskLoss {meter_maskloss.avg:.6f} "
                f"TotalLoss {meter_loss.avg:.4f} "
                f"Prec@1 {meter_top1.avg:.2f}"
            )

            masks = [round(m.mask.mean().item(), 2) for m in self.student.mask_modules]
            self.logger.info(f"[Train mask avg] Epoch {epoch} : {masks}")
            self.logger.info(
                f"[Train model Flops] Epoch {epoch} : {Flops.item() / (10**6)}M"
            )

            # Validation
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description(f"epoch: {epoch}/{self.num_epochs}")
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits_student, _ = self.student(images)
                        probs = torch.softmax(logits_student, dim=1)[:, 1].cpu().numpy()
                        preds = torch.argmax(logits_student, dim=1).cpu().numpy()
                        targets_np = targets.cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(targets_np)
                        all_probs.extend(probs)
                        prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)

                        _tqdm.set_postfix(top1="{:.4f}".format(meter_top1.avg))
                        _tqdm.update(1)

            Flops = self.student.get_flops()
            f1 = f1_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)
            cm = confusion_matrix(all_labels, all_preds)

            self.writer.add_scalar("val/acc/top1", meter_top1.avg, epoch)
            self.writer.add_scalar("val/Flops", Flops, epoch)
            self.writer.add_scalar("val/f1_score", f1, epoch)
            self.writer.add_scalar("val/auc_roc", auc, epoch)

            self.logger.info(
                f"[Val] Epoch {epoch} : Prec@1 {meter_top1.avg:.2f}, F1-Score {f1:.4f}, AUC-ROC {auc:.4f}"
            )
            self.logger.info(f"[Val Confusion Matrix] Epoch {epoch} :\n{cm}")
            self.logger.info(f"[Val mask avg] Epoch {epoch} : {masks}")
            self.logger.info(
                f"[Val model Flops] Epoch {epoch} : {Flops.item() / (10**6)}M"
            )

            # Early Stopping
            val_loss = meter_loss.avg
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    self.logger.info('Early stopping')
                    break

            self.start_epoch += 1
            if self.best_prec1 < meter_top1.avg:
                self.best_prec1 = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)

            self.logger.info(f" => Best top1 accuracy: {self.best_prec1}")

        self.logger.info("Training finished!")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()

# تنظیم آرگومان‌ها
class Args:
    dataset_dir = '/kaggle/input/hardfakevsrealfaces'
    dataset_type = 'hardfakevsreal'
    csv_file = '/kaggle/input/hardfakevsrealfaces/data.csv'  # مسیر فایل CSV
    num_workers = 4
    pin_memory = True
    arch = 'ResNet_50'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    result_dir = './results'
    teacher_ckpt_path = 'best_teacher.pth'  # مسیر مدل فاین‌تیون‌شده
    num_epochs = 50
    lr = 0.001
    warmup_steps = 5
    warmup_start_lr = 1e-5
    lr_decay_T_max = 50
    lr_decay_eta_min = 1e-5
    weight_decay = 1e-4
    train_batch_size = 32
    eval_batch_size = 32
    target_temperature = 4.0
    gumbel_start_temperature = 1.0
    gumbel_end_temperature = 0.1
    coef_kdloss = 1.0
    coef_rcloss = 1.0
    coef_maskloss = 0.01
    compress_rate = 0.5
    resume = None

# اجرای آموزش
args = Args()
trainer = Train(args)
trainer.main()
