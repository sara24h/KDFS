import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import utils, loss, meter, scheduler
from get_flops_and_params import get_flops_and_params

# تنظیمات محیطی برای جلوگیری از خطاهای CUDA
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # برای دیباگ خطاهای CUDA

class Finetune:
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
        self.finetune_train_batch_size = args.finetune_train_batch_size
        self.finetune_eval_batch_size = args.finetune_eval_batch_size
        self.finetune_student_ckpt_path = args.finetune_student_ckpt_path
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_lr = args.finetune_lr
        self.finetune_warmup_steps = args.finetune_warmup_steps
        self.finetune_warmup_start_lr = args.finetune_warmup_start_lr
        self.finetune_lr_decay_T_max = args.finetune_lr_decay_T_max
        self.finetune_lr_decay_eta_min = args.finetune_lr_decay_eta_min
        self.finetune_weight_decay = args.finetune_weight_decay
        self.finetune_resume = args.resume  # استفاده از resume به جای finetune_resume
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.start_epoch = 0
        self.best_prec1_after_finetune = 0

    def result_init(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
        self.logger = utils.get_logger(
            os.path.join(self.result_dir, "finetune_logger.log"), "finetune_logger"
        )
        self.logger.info("Finetune config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(
            self.args,
            os.path.join(self.result_dir, "finetune_config.txt")
        )
        self.logger.info("--------- Finetune -----------")

    def setup_seed(self):
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
        self.logger.info("==> Loading datasets..")
        if self.dataset_mode not in ['hardfake', 'rvf10k']:
            raise ValueError("dataset_mode must be 'hardfake' or 'rvf10k'")
        
        if self.dataset_mode == 'hardfake':
            hardfake_csv_file = os.path.join(self.dataset_dir, 'data.csv')
            hardfake_root_dir = self.dataset_dir
            rvf10k_train_csv = None
            rvf10k_valid_csv = None
            rvf10k_root_dir = None
        else:  # rvf10k
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
            train_batch_size=self.finetune_train_batch_size,
            eval_batch_size=self.finetune_eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=False
        )

        self.train_loader = dataset_instance.loader_train
        self.val_loader = dataset_instance.loader_test
        self.logger.info(f"{self.dataset_mode} dataset loaded! Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

    def build_model(self):
        self.logger.info("==> Building sparse student model for fine-tuning..")
        self.student = ResNet_50_sparse_hardfakevsreal()
        ckpt_student = torch.load(self.finetune_student_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
        self.student.load_state_dict(state_dict, strict=False)
        self.best_prec1_before_finetune = ckpt_student.get("best_prec1", 0.0)
        self.student.to(self.device)

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss()  # مناسب برای طبقه‌بندی باینری

    def define_optim(self):
        weight_params = [p for n, p in self.student.named_parameters() if p.requires_grad and "mask" not in n]
        self.finetune_optim_weight = torch.optim.Adamax(
            weight_params,
            lr=self.finetune_lr,
            weight_decay=self.finetune_weight_decay,
            eps=1e-7,
        )
        self.finetune_scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.finetune_optim_weight,
            T_max=self.finetune_lr_decay_T_max,
            eta_min=self.finetune_lr_decay_eta_min,
            warmup_steps=self.finetune_warmup_steps,
            warmup_start_lr=self.finetune_warmup_start_lr,
        )

    def resume_student_ckpt(self):
        ckpt_student = torch.load(self.finetune_resume, map_location="cpu", weights_only=True)
        self.best_prec1_after_finetune = ckpt_student["best_prec1_after_finetune"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.finetune_optim_weight.load_state_dict(ckpt_student["finetune_optim_weight"])
        self.finetune_scheduler_student_weight.load_state_dict(
            ckpt_student["finetune_scheduler_student_weight"]
        )
        self.logger.info(f"=> Continue from epoch {self.start_epoch}...")

    def save_student_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)
        ckpt_student = {
            "best_prec1_after_finetune": self.best_prec1_after_finetune,
            "start_epoch": self.start_epoch,
            "student": self.student.state_dict(),
            "finetune_optim_weight": self.finetune_optim_weight.state_dict(),
            "finetune_scheduler_student_weight": self.finetune_scheduler_student_weight.state_dict(),
        }
        if is_best:
            torch.save(
                ckpt_student,
                os.path.join(folder, f"finetune_{self.arch}_sparse_best.pt"),
            )
        if self.sparsed_student_ckpt_path:
            torch.save(
                ckpt_student,
                self.sparsed_student_ckpt_path
            )
        torch.save(
            ckpt_student,
            os.path.join(folder, f"finetune_{self.arch}_sparse_last.pt"),
        )

    def finetune(self):
        if self.device == "cuda":
            self.ori_loss = self.ori_loss.cuda()
            torch.cuda.empty_cache()

        if self.finetune_resume:
            self.resume_student_ckpt()

        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(self.start_epoch, self.finetune_num_epochs):
            self.student.train()
            self.student.ticket = True
            meter_oriloss.reset()
            meter_loss.reset()
            meter_top1.reset()
            finetune_lr = (
                self.finetune_optim_weight.state_dict()["param_groups"][0]["lr"]
                if epoch > 0
                else self.finetune_warmup_start_lr
            )

            with tqdm(total=len(self.train_loader), ncols=100, desc=f"Finetune Epoch {epoch+1}/{self.finetune_num_epochs}") as _tqdm:
                for images, targets in self.train_loader:
                    self.finetune_optim_weight.zero_grad()
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    logits_student, _ = self.student(images)
                    logits_student = logits_student.squeeze()  # تبدیل به (batch_size,)
                    ori_loss = self.ori_loss(logits_student, targets)
                    total_loss = ori_loss
                    total_loss.backward()
                    self.finetune_optim_weight.step()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100. * correct / images.size(0)
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1, n)

                    _tqdm.set_postfix(loss=f"{meter_loss.avg:.4f}", top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)
                    time.sleep(0.01)

            self.finetune_scheduler_student_weight.step()

            self.writer.add_scalar("finetune_train/loss/ori_loss", meter_oriloss.avg, epoch + 1)
            self.writer.add_scalar("finetune_train/loss/total_loss", meter_loss.avg, epoch + 1)
            self.writer.add_scalar("finetune_train/acc/top1", meter_top1.avg, epoch + 1)
            self.writer.add_scalar("finetune_train/lr/lr", finetune_lr, epoch + 1)

            self.logger.info(
                f"[Finetune_train] Epoch {epoch+1}: "
                f"LR {finetune_lr:.6f} "
                f"OriLoss {meter_oriloss.avg:.4f} "
                f"TotalLoss {meter_loss.avg:.4f} "
                f"Prec@1 {meter_top1.avg:.2f}"
            )

            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100, desc=f"Finetune Val Epoch {epoch+1}/{self.finetune_num_epochs}") as _tqdm:
                    for images, targets in self.val_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze()
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100. * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)
                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            self.writer.add_scalar("finetune_val/acc/top1", meter_top1.avg, epoch + 1)

            self.logger.info(
                f"[Finetune_val] Epoch {epoch+1}: Prec@1 {meter_top1.avg:.2f}"
            )

            masks = [round(m.mask.mean().item(), 2) for m in self.student.mask_modules]
            self.logger.info(f"[Finetune Mask avg] Epoch {epoch+1}: {masks}")

            self.start_epoch = epoch + 1
            if self.best_prec1_after_finetune < meter_top1.avg:
                self.best_prec1_after_finetune = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)

            self.logger.info(
                f" => Best top1 accuracy before finetune: {self.best_prec1_before_finetune:.2f}"
            )
            self.logger.info(
                f" => Best top1 accuracy after finetune: {self.best_prec1_after_finetune:.2f}"
            )

        self.logger.info("Finetune finished!")
        self.logger.info(f"Best top1 accuracy: {self.best_prec1_after_finetune:.2f}")
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(self.args)
        self.logger.info(
            f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
            f"Params reduction: {Params_reduction:.2f}%"
        )
        self.logger.info(
            f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
            f"Flops reduction: {Flops_reduction:.2f}%"
        )

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.finetune()
