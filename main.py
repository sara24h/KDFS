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
from torch.amp import GradScaler, autocast
from data.dataset import Dataset_hardfakevsreal, FaceDataset  
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import utils, loss, meter, scheduler
import json
import time
from test import Test
from train import Train
from finetune import Finetune

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

matplotlib.use('Agg')



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
        choices=["hardfakevsreal", "FakeVsReal10kDataset"],
        help="Type of dataset: hardfakevsreal or FakeVsReal10kDataset"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/kaggle/input/hardfakevsrealfaces/data.csv",
        help="The path to the CSV file",
    )
    parser.add_argument(
        "--train_csv_file",
        type=str,
        default="/kaggle/input/rvf10k/train.csv",
        help="Path to train CSV file (for FakeVsReal10kDataset)"
    )
    parser.add_argument(
        "--valid_csv_file",
        type=str,
        default="/kaggle/input/rvf10k/valid.csv",
        help="Path to valid CSV file (for FakeVsReal10kDataset)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=32,
        help="Batch size for testing"
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

def load_dataset(args):
    if args.dataset_type == "hardfakevsreal":
        if args.csv_file is None:
            raise ValueError("CSV file path is required for hardfakevsreal dataset")
        dataset = Dataset_hardfakevsreal(
            csv_file=args.csv_file,
            root_dir=args.dataset_dir,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            ddp=args.ddp
        )
        train_loader = dataset.loader_train
        valid_loader = dataset.loader_test  # در این دیتاست، loader_test برای اعتبارسنجی است
        test_loader = None
    elif args.dataset_type == "FakeVsReal10kDataset":
        if args.train_csv_file is None or args.valid_csv_file is None:
            raise ValueError("Train and valid CSV files are required for FakeVsReal10kDataset")
        dataset = FakeVsReal10kDataset(
            train_csv_file=args.train_csv_file,
            valid_csv_file=args.valid_csv_file,
            root_dir=args.dataset_dir,
            train_batch_size=args.train_batch_size,
            valid_batch_size=args.eval_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            ddp=args.ddp
        )
        train_loader = dataset.loader_train
        valid_loader = dataset.loader_valid
        test_loader = dataset.loader_test
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    return train_loader, valid_loader, test_loader

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
