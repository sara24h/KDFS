import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast  # به‌روزرسانی به torch.amp
from data.dataset import Dataset_hardfakevsreal
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import utils, loss, meter, scheduler
import json
import time
from test import Test
# تنظیم متغیر محیطی برای مدیریت حافظه
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# تنظیم backend غیرتعاملی برای ذخیره تصاویر
matplotlib.use('Agg')

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
        self.phase = args.phase
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.csv_file = args.csv_file
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs if self.phase == 'train' else args.finetune_num_epochs
        self.lr = args.lr if self.phase == 'train' else args.finetune_lr
        self.warmup_steps = args.warmup_steps if self.phase == 'train' else args.finetune_warmup_steps
        self.warmup_start_lr = args.warmup_start_lr if self.phase == 'train' else args.finetune_warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max if self.phase == 'train' else args.finetune_lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min if self.phase == 'train' else args.finetune_lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size if self.phase == 'train' else args.finetune_train_batch_size
        self.eval_batch_size = args.eval_batch_size if self.phase == 'train' else args.finetune_eval_batch_size
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        self.compress_rate = args.compress_rate
        self.resume = args.finetune_student_ckpt_path if self.phase == 'finetune' else None
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path if self.phase == 'finetune' else None
        self.start_epoch = 0
        self.best_prec1 = 0

    def result_init(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
        self.logger = utils.get_logger(os.path.join(self.result_dir, f"{self.phase}_logger.log"), f"{self.phase}_logger")
        self.logger.info(f"{self.phase} config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(self.args, os.path.join(self.result_dir, f"{self.phase}_config.txt"))
        self.logger.info(f"--------- {self.phase.capitalize()} -----------")

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
        dataset_class = globals()["Dataset_" + self.dataset_type]
        train_loader, val_loader, _ = dataset_class.get_loaders(
            data_dir=self.dataset_dir,
            csv_file=self.csv_file,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=self.args.ddp
        )
        self.train_loader, self.val_loader = train_loader, val_loader
        df = pd.read_csv(self.csv_file)
        train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
        test_csv_file = os.path.join(self.result_dir, 'test_data.csv')
        test_df.to_csv(test_csv_file, index=False)
        val_test_transform = dataset_class.get_val_test_transform()
        test_dataset = dataset_class(self.dataset_dir, test_csv_file, transform=val_test_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.test_df = test_df
        self.val_test_transform = val_test_transform
        self.logger.info("Dataset and test loader have been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")
        if self.phase == 'train':
            self.logger.info("Loading teacher model")
            self.teacher = eval(self.arch + "_" + "hardfakevsreal")()
            if not os.path.exists(self.teacher_ckpt_path):
                raise FileNotFoundError(f"Teacher checkpoint not found at: {self.teacher_ckpt_path}")
            ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu", weights_only=True)
            if self.arch in ["resnet_56", "resnet_110", "VGG_16_bn", "DenseNet_40", "GoogLeNet"]:
                self.teacher.load_state_dict(ckpt_teacher["state_dict"])
            elif self.arch in ["ResNet_18", "ResNet_50", "MobileNetV2"]:
                try:
                    self.teacher.load_state_dict(ckpt_teacher, strict=True)
                except RuntimeError as e:
                    self.logger.error(f"Error loading teacher state_dict: {e}")
                    raise
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
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )
        self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
            self.optim_mask,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )
        

    def resume_student_ckpt(self):
        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.logger.info("=> Continue from epoch {}...".format(self.start_epoch))

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
        }
        if self.phase == 'train':
            if is_best:
                torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_best.pt"))
            torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))
        else:  # finetune
            if is_best:
                torch.save(ckpt_student, self.sparsed_student_ckpt_path)
            torch.save(ckpt_student, os.path.join(folder, f"finetune_{self.arch}_sparse_last.pt"))

    

    def train(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            if self.phase == 'train':
                self.teacher = self.teacher.cuda()
            self.student = self.student.cuda()
            self.ori_loss = self.ori_loss.cuda()
            self.kd_loss = self.kd_loss.cuda()
            self.rc_loss = self.rc_loss.cuda()
            self.mask_loss = self.mask_loss.cuda()
        if self.resume:
            self.resume_student_ckpt()
        scaler = GradScaler('cuda')
        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
        meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
        meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        if self.phase == 'train':
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
                        targets = targets.cuda()
                    with autocast('cuda'):
                        logits_student, feature_list_student = self.student(images)
                        if self.phase == 'train':
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
                        else:  # finetune
                            ori_loss = self.ori_loss(logits_student, targets)
                            total_loss = ori_loss
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optim_weight)
                    scaler.step(self.optim_mask)
                    scaler.update()
                    prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    if self.phase == 'train':
                        meter_kdloss.update(self.coef_kdloss * kd_loss.item(), n)
                        meter_rcloss.update(self.coef_rcloss * rc_loss.item() / len(feature_list_student), n)
                        meter_maskloss.update(self.coef_maskloss * mask_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)
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
            if self.phase == 'train':
                self.writer.add_scalar("train/loss/kd_loss", meter_kdloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/rc_loss", meter_rcloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/mask_loss", meter_maskloss.avg, global_step=epoch)
            self.writer.add_scalar("train/loss/total_loss", meter_loss.avg, global_step=epoch)
            self.writer.add_scalar("train/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("train/lr/lr", lr, global_step=epoch)
            self.writer.add_scalar("train/temperature/gumbel_temperature", self.student.gumbel_temperature, global_step=epoch)
            self.writer.add_scalar("train/Flops", Flops, global_step=epoch)
            self.logger.info(
                f"[{self.phase.capitalize()}] "
                f"Epoch {epoch} : "
                f"Gumbel_temperature {self.student.gumbel_temperature:.2f} "
                f"LR {lr:.6f} "
                f"OriLoss {meter_oriloss.avg:.4f} "
                + (f"KDLoss {meter_kdloss.avg:.4f} " if self.phase == 'train' else "")
                + (f"RCLoss {meter_rcloss.avg:.4f} " if self.phase == 'train' else "")
                + (f"MaskLoss {meter_maskloss.avg:.6f} " if self.phase == 'train' else "")
                + f"TotalLoss {meter_loss.avg:.4f} "
                + f"Prec@1 {meter_top1.avg:.2f}"
             )

            
            masks = [round(m.mask.mean().item(), 2) for _, m in enumerate(self.student.mask_modules)]
            self.logger.info(f"[{self.phase.capitalize()} mask avg] Epoch {epoch} : {masks}")
            self.logger.info(f"[{self.phase.capitalize()} model Flops] Epoch {epoch} : {Flops.item() / 1e6:.2f}M")
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        with autocast('cuda'):
                            logits_student, _ = self.student(images)
                        prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        _tqdm.set_postfix(top1="{:.4f}".format(meter_top1.avg))
                        _tqdm.update(1)
                        time.sleep(0.01)
            Flops = self.student.get_flops()
            self.writer.add_scalar("val/acc/top1", meter_top1.avg, global_step=epoch)
            self.writer.add_scalar("val/Flops", Flops, global_step=epoch)
            self.logger.info(f"[Val] Epoch {epoch} : Prec@1 {meter_top1.avg:.2f}")

            masks = [round(m.mask.mean().item(), 2) for _, m in enumerate(self.student.mask_modules)]
            self.logger.info(f"[Val mask avg] Epoch {epoch} : {masks}")
            self.logger.info(f"[Val model Flops] Epoch {epoch} : {Flops.item() / 1e6:.2f}M")
            self.start_epoch += 1
            if self.best_prec1 < meter_top1.avg:
                self.best_prec1 = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)
            self.logger.info(f" => Best top1 accuracy before {'finetune' if self.phase == 'train' else 'final'} : {self.best_prec1}")
        self.logger.info(f"{self.phase.capitalize()} finished!")
        

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()

class Finetune(Train):
    def __init__(self, args):
        super().__init__(args)

def parse_args():
    desc = "Pytorch implementation of KDFS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=("train", "finetune", "test"),
        help="train, finetune or test",
    )

    parser.add_argument(
        "--dataset_dir", type=str, default="/kaggle/input/hardfakevsrealfaces", help="The dataset path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="hardfakevsreal",
        choices=("cifar10", "cifar100", "imagenet","hardfakevsreal"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/kaggle/input/hardfakevsrealfaces/data.csv",
        help="The path to the CSV file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The num_workers of dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="The pin_memory of dataloader",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="load the model from the specified checkpoint",
    )
    parser.add_argument("--ddp", action="store_true", help="Use the distributed data parallel")
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=(
            "ResNet_18",
            "ResNet_50",
            "VGG_16_bn",
            "resnet_56",
            "resnet_110",
            "DenseNet_40",
            "GoogLeNet",
            "MobileNetV2",
        ),
        help="The architecture to prune",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Init seed")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./result/",
        help="The directory where the results will be stored",
    )
    parser.add_argument(
        "--dali",
        action="store_true",
        help="Use dali",
    )

    parser.add_argument(
        "--teacher_ckpt_path",
        type=str,
        default="/kaggle/working/KDFS/teacher_dir/teacher_model.pth",
        help="The path where the teacher model is stored",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="The num of epochs to train."
    )
    parser.add_argument(
        "--lr", default=5e-4, type=float, help="The initial learning rate of model"
    )
    parser.add_argument(
        "--warmup_steps",
        default=30,
        type=int,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-4,
        type=float,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--lr_decay_T_max",
        default=350,
        type=int,
        help="T_max of CosineAnnealingLR",
    )
    parser.add_argument(
        "--lr_decay_eta_min",
        default=5e-6,
        type=float,
        help="eta_min of CosineAnnealingLR",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for validation"
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=3,
        help="temperature of soft targets",
    )
    parser.add_argument(
        "--gumbel_start_temperature",
        type=float,
        default=2,
        help="Gumbel-softmax temperature at the start of training",
    )
    parser.add_argument(
        "--gumbel_end_temperature",
        type=float,
        default=0.1,
        help="Gumbel-softmax temperature at the end of training",
    )
    parser.add_argument(
        "--coef_kdloss", type=float, default=0.5, help="Coefficient of kd loss"
    )
    parser.add_argument(
        "--coef_rcloss",
        type=float,
        default=100,
        help="Coefficient of reconstruction loss",
    )
    parser.add_argument(
        "--coef_maskloss", type=float, default=1.0, help="Coefficient of mask loss"
    )
    parser.add_argument(
        "--compress_rate",
        type=float,
        default=0.68,
        help="Compress rate of the student model",
    )

    parser.add_argument(
        "--finetune_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the student ckpt in finetune",
    )
    parser.add_argument(
        "--finetune_num_epochs",
        type=int,
        default=20,
        help="The num of epochs to train in finetune",
    )
    parser.add_argument(
        "--finetune_lr",
        default=4e-6,
        type=float,
        help="The initial learning rate of model in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_steps",
        default=5,
        type=int,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_start_lr",
        default=4e-8,
        type=float,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_T_max",
        default=20,
        type=int,
        help="T_max of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_eta_min",
        default=4e-8,
        type=float,
        help="eta_min of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=2e-5,
        help="Weight decay in finetune",
    )
    parser.add_argument(
        "--finetune_train_batch_size",
        type=int,
        default=16,
        help="Batch size for training in finetune",
    )
    parser.add_argument(
        "--finetune_eval_batch_size",
        type=int,
        default=16,
        help="Batch size for validation in finetune",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to save the sparsed student ckpt in finetune",
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size for test"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    if args.ddp:
        raise NotImplementedError("Distributed Data Parallel (DDP) is not implemented in this version.")
    else:
        if args.phase == "train":
            train = Train(args=args)
            train.main()
        elif args.phase == "finetune":
            finetune = Finetune(args=args)
            finetune.main()
        elif args.phase == "test":
            test = Test(args=args)
            test.main()


if __name__ == "__main__":
    main()
