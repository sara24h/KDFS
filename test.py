import os
import time
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k, ResNet_50_sparse_140k
from utils import utils, meter
from get_flops_and_params import get_flops_and_params

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch  # باید 'ResNet_50' باشد
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode  # 'hardfake', 'rvf10k', یا '140k'

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            if self.dataset_mode == 'hardfake':
                dataset = Dataset_selector(
                    dataset_mode='hardfake',
                    hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv'),
                    hardfake_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False,
                    max_samples=140000  # محدود کردن به 140k نمونه (در صورت نیاز)
                )
            elif self.dataset_mode == 'rvf10k':
                dataset = Dataset_selector(
                    dataset_mode='rvf10k',
                    rvf10k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                    rvf10k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                    rvf10k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False,
                    max_samples=140000  # محدود کردن به 140k نمونه (در صورت نیاز)
                )
            else:  # فرضاً حالت 140k
                dataset = Dataset_selector(
                    dataset_mode='140k',
                    hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv'),  # یا فایل مناسب
                    hardfake_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False,
                    max_samples=140000  # محدود کردن به 140k نمونه
                )
            
            self.test_loader = dataset.loader_test
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self):
        print("==> Building student model..")
        try:
            print("Loading sparse student model")
            if self.dataset_mode == 'hardfake':
                self.student = ResNet_50_sparse_hardfakevsreal()
            elif self.dataset_mode == 'rvf10k':
                self.student = ResNet_50_sparse_rvf10k()
            else:  # فرضاً برای 140k
                self.student = ResNet_50_sparse_140k()  # مدل جدید با حدود 140k پارامتر
            
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            self.student.load_state_dict(state_dict, strict=True)
            self.student.to(self.device)
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.student.eval()
        self.student.ticket = True  # فعال کردن حالت ticket برای مدل sparse
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc="Test") as _tqdm:
                    for images, targets in self.test_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()  # تبدیل به float برای طبقه‌بندی باینری
                        
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze()  # تبدیل به (batch_size,) برای طبقه‌بندی باینری
                        preds = (torch.sigmoid(logits_student) > 0.5).float()  # پیش‌بینی با sigmoid و آستانه 0.5
                        correct = (preds == targets).sum().item()
                        prec1 = 100. * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            print(f"[Test] Dataset: {self.dataset_mode}, Prec@1: {meter_top1.avg:.2f}%")

            # محاسبه FLOPs و پارامترها
            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = get_flops_and_params(args=self.args)
            print(
                f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
                f"Params reduction: {Params_reduction:.2f}%"
            )
            print(
                f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"Flops reduction: {Flops_reduction:.2f}%"
            )
            # بررسی تعداد پارامترها برای 140k
            if abs(Params * 1e6 - 140000) > 10000:  # اگر تعداد پارامترها بیش از 10k از 140k فاصله داشته باشد
                print(f"Warning: Model parameters ({Params:.2f}M) are not close to 140k!")
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

    def main(self):
        try:
            self.dataload()
            self.build_model()
            self.test()
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise
