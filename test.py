import os
import time
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import Dataset_selector  # استفاده از Dataset_selector به جای Dataset_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
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
        self.dataset_mode = args.dataset_mode  # اضافه کردن dataset_mode برای انتخاب 'hardfake' یا 'rvf10k'

    def dataload(self):
        # استفاده از Dataset_selector برای بارگذاری داده‌ها
        if self.dataset_mode == 'hardfake':
            dataset = Dataset_selector(
                dataset_mode='hardfake',
                hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv'),
                hardfake_root_dir=self.dataset_dir,
                train_batch_size=self.test_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False
            )
        else:  # rvf10k
            dataset = Dataset_selector(
                dataset_mode='rvf10k',
                rvf10k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                rvf10k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                rvf10k_root_dir=self.dataset_dir,
                train_batch_size=self.test_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False
            )
        
        self.val_loader = dataset.loader_test  # استفاده از loader_test برای اعتبارسنجی
        print(f"{self.dataset_mode} validation dataset has been loaded! Total batches: {len(self.val_loader)}")

    def build_model(self):
        print("==> Building student model..")
        print("Loading sparse student model")
        self.student = ResNet_50_sparse_hardfakevsreal()  # مقداردهی اولیه بدون پارامترهای Gumbel برای تست
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        # بررسی وجود کلید 'student' در چک‌پوینت
        state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
        self.student.load_state_dict(state_dict)
        self.student.to(self.device)  # انتقال مدل به دستگاه مناسب در اینجا
        print(f"Model loaded on {self.device}")

    def test(self):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.student.eval()
        self.student.ticket = True  # فعال کردن حالت ticket برای مدل sparse
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), ncols=100, desc="Testing") as _tqdm:
                for images, targets in self.val_loader:
                    images = images.to(self.device, non_blocking=True)  # بهینه‌سازی انتقال داده
                    targets = targets.to(self.device, non_blocking=True)
                    
                    logits_student, _ = self.student(images)
                    prec1 = utils.get_accuracy(logits_student, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_top1.update(prec1.item(), n)

                    _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)
                    time.sleep(0.01)  # حفظ برای جلوگیری از بارگذاری بیش از حد رابط کاربری

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

    def main(self):
        self.dataload()
        self.build_model()
        self.test()
