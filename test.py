###TEST###

import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, loss, meter, scheduler
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from get_flops_and_params import get_flops_and_params

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"fake": 0, "real": 1}  # Fake=0, Real=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx]['images_id']}.jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.label_map[self.data.iloc[idx]["label"]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

    def dataload(self):
        print(f"Loading {self.dataset_type} dataset...")
        
        # تعریف transform برای تست
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.dataset_type == "hardfakevsrealfaces":
            csv_file = os.path.join(self.dataset_dir, "data.csv")
            img_dir = os.path.join(self.dataset_dir, "images")
            dataset = CustomDataset(csv_file=csv_file, img_dir=img_dir, transform=transform_test)
            # فقط داده‌های تست (validation) رو لود می‌کنیم
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.test_batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        print(f"{self.dataset_type} dataset has been loaded!")

    def build_model(self):
        print("==> Building model..")

        print("Loading student model")
        if self.arch == "ResNet_50":
            # پارامترهای Gumbel رو باید با مقادیر استفاده‌شده در آموزش سازگار کنیم
            self.student = ResNet_50_sparse_imagenet(
                gumbel_start_temperature=1.0,  # مقدار پیش‌فرض، باید با آموزش مطابقت داشته باشه
                gumbel_end_temperature=0.1,    # مقدار پیش‌فرض
                num_epochs=50                  # مقدار پیش‌فرض، باید با آموزش مطابقت داشته باشه
            )
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
            self.student.load_state_dict(ckpt_student["student"])
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")

    def test(self):
        if self.device == "cuda":
            self.student = self.student.cuda()

        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        self.student.eval()
        self.student.ticket = True  # فعال کردن حالت فشرده‌شده
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                for images, targets in self.val_loader:
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits_student, _ = self.student(images)
                    prec1, prec5 = utils.get_accuracy(logits_student, targets, topk=(1, 5))
                    n = images.size(0)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)

                    _tqdm.set_postfix(
                        top1="{:.4f}".format(meter_top1.avg),
                        top5="{:.4f}".format(meter_top5.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

        print(
            "[Test] Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                top1=meter_top1.avg,
                top5=meter_top5.avg,
            )
        )

        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(args=self.args)
        print(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        print(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

    def main(self):
        self.dataload()
        self.build_model()
        self.test()
