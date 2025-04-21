import argparse
import os
from train import Train
from finetune import Finetune  # Assuming your fine-tuning script is named fine_tune.py

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from data.dataset import Dataset_hardfakevsreal
from utils import utils, meter, scheduler

class TrainTeacher:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.csv_file = args.csv_file
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
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

        self.best_prec1 = 0

    def result_init(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.teacher_dir):
            os.makedirs(self.teacher_dir)

        self.writer = SummaryWriter(self.result_dir)
        self.logger = utils.get_logger(
            os.path.join(self.result_dir, "teacher_train_logger.log"), "teacher_train_logger"
        )
        self.logger.info("Teacher train config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(
            self.args, os.path.join(self.result_dir, "teacher_train_config.txt")
        )
        self.logger.info("--------- Teacher Training -----------")

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
            self.csv_file,
            self.train_batch_size,
            self.eval_batch_size,
            self.num_workers,
            self.pin_memory,
            ddp=False
        )

        # مجموعه تست (10% از داده‌ها)
        dataset = Dataset_hardfakevsreal(self.dataset_dir, self.csv_file, transform=transforms.Compose([
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        test_size = int(0.1 * len(dataset))
        train_val_size = len(dataset) - test_size
        _, test_dataset = torch.utils.data.random_split(dataset, [train_val_size, test_size])
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        self.logger.info("HardFakeVsRealFaces dataset loaded! Train: {}, Val: {}, Test: {}".format(
            len(self.train_loader.dataset), len(self.val_loader.dataset), len(self.test_loader.dataset)))

    def build_model(self):
        self.logger.info("==> Building teacher model (ResNet50)")
        self.teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.teacher.fc.in_features
        self.teacher.fc = nn.Linear(num_ftrs, 2)  # 2 کلاس: Fake و Real
        self.teacher = self.teacher.to(self.device)

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss().to(self.device)

    def define_optim(self):
        base_params = [p for n, p in self.teacher.named_parameters() if 'fc' not in n]
        fc_params = [p for n, p in self.teacher.named_parameters() if 'fc' in n]
        self.optim = torch.optim.SGD([
            {'params': base_params, 'lr': self.lr / 10},
            {'params': fc_params, 'lr': self.lr}
        ], momentum=0.9, weight_decay=self.weight_decay)

        self.scheduler = scheduler.CosineAnnealingLRWarmup(
            self.optim,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )

    def save_teacher_ckpt(self, is_best, epoch):
        ckpt = {
            'epoch': epoch,
            'best_prec1': self.best_prec1,
            'state_dict': self.teacher.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(ckpt, os.path.join(self.teacher_dir, 'teacher_last.pt'))
        if is_best:
            torch.save(ckpt, os.path.join(self.teacher_dir, 'teacher_best.pt'))

    def finetune_teacher(self):
        self.teacher.train()
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(1, self.num_epochs + 1):
            meter_loss.reset()
            meter_top1.reset()

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description(f"Teacher Finetune epoch: {epoch}/{self.num_epochs}")
                for images, targets in self.train_loader:
                    self.optim.zero_grad()
                    images, targets = images.to(self.device), targets.to(self.device)

                    logits = self.teacher(images)
                    loss = self.ori_loss(logits, targets)
                    loss.backward()
                    self.optim.step()

                    prec1 = utils.get_accuracy(logits, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_loss.update(loss.item(), n)
                    meter_top1.update(prec1.item(), n)

                    _tqdm.set_postfix(loss=f"{meter_loss.avg:.4f}", top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)

            self.scheduler.step()
            self.writer.add_scalar("train/loss", meter_loss.avg, epoch)
            self.writer.add_scalar("train/acc/top1", meter_top1.avg, epoch)
            self.writer.add_scalar("train/lr", self.optim.param_groups[0]['lr'], epoch)

            self.logger.info(
                f"[Train] Epoch {epoch}: Loss {meter_loss.avg:.4f}, Acc@1 {meter_top1.avg:.2f}"
            )

            # اعتبارسنجی
            self.teacher.eval()
            meter_top1.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description(f"Validation epoch: {epoch}/{self.num_epochs}")
                    for images, targets in self.val_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        logits = self.teacher(images)
                        prec1 = utils.get_accuracy(logits, targets, topk=(1,))[0]
                        meter_top1.update(prec1.item(), images.size(0))
                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)

            self.writer.add_scalar("val/acc/top1", meter_top1.avg, epoch)
            self.logger.info(f"[Val] Epoch {epoch}: Acc@1 {meter_top1.avg:.2f}")

            is_best = meter_top1.avg > self.best_prec1
            if is_best:
                self.best_prec1 = meter_top1.avg
            self.save_teacher_ckpt(is_best, epoch)

        self.logger.info(f"Teacher finetuning finished! Best Acc@1: {self.best_prec1}")

        # تست
        self.teacher.eval()
        meter_top1.reset()
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=100) as _tqdm:
                _tqdm.set_description("Test")
                for images, targets in self.test_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    logits = self.teacher(images)
                    prec1 = utils.get_accuracy(logits, targets, topk=(1,))[0]
                    meter_top1.update(prec1.item(), images.size(0))
                    _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)

        self.writer.add_scalar("test/acc/top1", meter_top1.avg, self.num_epochs)
        self.logger.info(f"[Test] Acc@1: {meter_top1.avg:.2f}")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.finetune_teacher()

class Args:
    dataset_dir = '/kaggle/input/hardfakevsrealfaces'
    csv_file = '/kaggle/input/hardfakevsrealfaces/data.csv'  # فرض بر وجود فایل متادیتا
    num_workers = 4
    pin_memory = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    result_dir = './results'
    teacher_dir = './teacher_dir'
    num_epochs = 50
    lr = 0.001
    warmup_steps = 5
    warmup_start_lr = 0.0001
    lr_decay_T_max = 50
    lr_decay_eta_min = 0.00001
    weight_decay = 5e-4
    train_batch_size = 32
    eval_batch_size = 32

if __name__ == "__main__":
    args = Args()
    trainer = TrainTeacher(args)
    trainer.main()













def parse_args():
    desc = "Pytorch implementation of KDFS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=("train", "finetune"),
        help="train or finetune",
    )

    # Common
    parser.add_argument(
        "--dataset_dir", type=str, default="/kaggle/input/hardfakevsrealfaces", help="The dataset path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="hardfakevsrealfaces",
        choices=("cifar10", "hardfakevsrealfaces"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,  # Reduced for memory efficiency
        help="The num_workers of dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=False,  # Disabled for memory efficiency
        help="The pin_memory of dataloader",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=("ResNet_50", "resnet_56"),
        help="The architecture to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Init seed")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/",
        help="The directory where the results will be stored",
    )

    # Train
    parser.add_argument(
        "--teacher_ckpt_path",
        type=str,
        default="teacher_resnet50_finetuned.pth",
        help="The path where to load the teacher ckpt",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="The num of epochs to train"
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="The initial learning rate of model"
    )
    parser.add_argument(
        "--warmup_steps",
        default=5,
        type=int,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-6,
        type=float,
        help="The starting learning rate for warmup",
    )
    parser.add_argument(
        "--lr_decay_T_max",
        default=10,
        type=int,
        help="T_max of CosineAnnealingLR",
    )
    parser.add_argument(
        "--lr_decay_eta_min",
        default=1e-6,
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
        "--accumulation_steps", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=4.0,
        help="Temperature of soft targets",
    )
    parser.add_argument(
        "--gumbel_start_temperature",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature at the start of training",
    )
    parser.add_argument(
        "--gumbel_end_temperature",
        type=float,
        default=0.1,
        help="Gumbel-softmax temperature at the end of training",
    )
    parser.add_argument(
        "--coef_kdloss", type=float, default=1.0, help="Coefficient of kd loss"
    )
    parser.add_argument(
        "--coef_rcloss",
        type=float,
        default=0.0,
        help="Coefficient of reconstruction loss",
    )
    parser.add_argument(
        "--coef_maskloss", type=float, default=0.0, help="Coefficient of mask loss"
    )
    parser.add_argument(
        "--compress_rate",
        type=float,
        default=0.5,
        help="Compress rate of the student model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load the model from the specified checkpoint",
    )

    # Finetune
    parser.add_argument(
        "--finetune_num_epochs",
        type=int,
        default=10,
        help="The num of epochs to train in finetune",
    )
    parser.add_argument(
        "--finetune_lr",
        default=1e-4,
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
        default=1e-6,
        type=float,
        help="The starting learning rate for warmup in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_T_max",
        default=10,
        type=int,
        help="T_max of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_eta_min",
        default=1e-6,
        type=float,
        help="eta_min of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=1e-4,
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
        "--finetune_resume",
        type=str,
        default=None,
        help="Load the model from the specified checkpoint in finetune",
    )
    parser.add_argument(
    "--csv_file",
    type=str,
    default="/kaggle/input/hardfakevsrealfaces/train.csv",
    help="Path to the CSV file for training/validation"
    )
    parser.add_argument(
        "--test_csv_file",
        type=str,
        default=None,
        help="Path to the CSV file for testing (optional)"
    )
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default="./teacher_dir",
        help="The directory where the trained teacher model will be saved" 
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        choices=("cifar10", "cifar100", "imagenet", "hardfakevsrealfaces"),
        help="The type of dataset"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "4"
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


    if args.phase == "train":
        train = Train(args=args)
        train.main()
    elif args.phase == "finetune":
        finetune = Finetune(args=args)
        finetune.main()
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")

if __name__ == "__main__":
    main()


