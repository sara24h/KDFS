import os
import time
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from model.student.MobileNetV2_sparse import MobileNetV2_sparse
from utils import utils, meter
from get_flops_and_params import get_flops_and_params
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode

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
                    ddp=False
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
                    ddp=False
                )
            elif self.dataset_mode == '140k':
                dataset = Dataset_selector(
                    dataset_mode='140k',
                    realfake140k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                    realfake140k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                    realfake140k_test_csv=os.path.join(self.dataset_dir, 'test.csv'),
                    realfake140k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            else:
                raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

            self.test_loader = dataset.loader_test
            self.test_dataset = dataset.test_dataset
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self, model_name):
        print(f"==> Building {model_name} student model..")
        try:
            print(f"Loading sparse {model_name} student model")
            ckpt_path = (
                self.sparsed_student_ckpt_path if model_name.lower() in self.sparsed_student_ckpt_path.lower()
                else self.sparsed_student_ckpt_path.replace("resnet_50", model_name.lower()).replace("mobilenetv2", model_name.lower())
            )
            if model_name == "ResNet_50":
                if self.dataset_mode == 'hardfake':
                    student = ResNet_50_sparse_hardfakevsreal()
                else:
                    student = ResNet_50_sparse_rvf10k()
            else:
                student = MobileNetV2_sparse(num_classes=1)

            ckpt_student = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            student.load_state_dict(state_dict, strict=True)
            student.to(self.device)
            print(f"{model_name} model loaded on {self.device}")
            return student
        except Exception as e:
            print(f"Error building {model_name} model: {str(e)}")
            raise

    def test_model(self, student, model_name):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        samples = []  # لیست برای ذخیره 20 نمونه
        sample_count = 0

        student.eval()
        student.ticket = True
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc=f"{model_name} Test") as _tqdm:
                    for batch_idx, (images, targets) in enumerate(self.test_loader):
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()

                        logits_student, _ = student(images)
                        logits_student = logits_student.squeeze()
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100. * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        # جمع‌آوری اطلاعات نمونه‌ها
                        for i in range(images.size(0)):
                            if sample_count < 20:
                                # فرض می‌کنیم دیتاست مسیر تصاویر را ارائه می‌دهد
                                img_path = self.test_dataset.samples[batch_idx * self.test_batch_size + i][0]
                                true_label = targets[i].item()
                                pred_label = preds[i].item()
                                samples.append({
                                    'image_path': img_path,
                                    'true_label': true_label,
                                    'predicted_label': pred_label
                                })
                                sample_count += 1

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

                        if sample_count >= 20:
                            break

            print(f"[{model_name} Test] Dataset: {self.dataset_mode}, Prec@1: {meter_top1.avg:.2f}%")

            # ذخیره نمونه‌ها در فایل CSV
            samples_df = pd.DataFrame(samples)
            samples_df.to_csv(f"/kaggle/working/{model_name}_test_samples.csv", index=False)
            print(f"20 test samples saved to /kaggle/working/{model_name}_test_samples.csv")

            # چاپ نمونه‌ها
            print("\n20 Test Samples:")
            for idx, sample in enumerate(samples):
                print(f"Sample {idx+1}:")
                print(f"  Image Path: {sample['image_path']}")
                print(f"  True Label: {sample['true_label']} (0: Real, 1: Fake)")
                print(f"  Predicted Label: {sample['predicted_label']} (0: Real, 1: Fake)")
                print()

            # نمایش تصاویر
            plt.figure(figsize=(15, 20))
            for idx, sample in enumerate(samples):
                try:
                    img = Image.open(sample['image_path'])
                    plt.subplot(5, 4, idx + 1)
                    plt.imshow(img)
                    plt.title(f"True: {sample['true_label']}\nPred: {sample['predicted_label']}")
                    plt.axis('off')
                except Exception as e:
                    print(f"Error displaying image {sample['image_path']}: {str(e)}")
            plt.tight_layout()
            plt.savefig(f"/kaggle/working/{model_name}_test_samples.png")
            plt.show()
            print(f"Images saved to /kaggle/working/{model_name}_test_samples.png")

            # محاسبه FLOPs و پارامترها
            original_arch = self.args.arch
            self.args.arch = model_name
            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = get_flops_and_params(args=self.args)
            self.args.arch = original_arch
            print(
                f"[{model_name}] Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
                f"Params reduction: {Params_reduction:.2f}%"
            )
            print(
                f"[{model_name}] Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"Flops reduction: {Flops_reduction:.2f}%"
            )
        except Exception as e:
            print(f"Error during {model_name} testing: {str(e)}")
            raise

    def test(self):
        # فقط ResNet_50 برای سادگی
        student_resnet = self.build_model("ResNet_50")
        self.test_model(student_resnet, "ResNet_50")

        # اگر MobileNetV2 هم نیاز است، این بخش را فعال کنید
        # student_mobile = self.build_model("MobileNetV2")
        # self.test_model(student_mobile, "MobileNetV2")

    def main(self):
        try:
            self.dataload()
            self.test()
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise

