import os
import time
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from utils import utils, meter
from get_flops_and_params import get_flops_and_params

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
        self.n_folds = args.n_folds if hasattr(args, 'n_folds') else 5  # تعداد فولدها
        self.result_dir = args.result_dir  # مسیر ذخیره نتایج فاین‌تیون

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
            self.test_loader = dataset.loader_test
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self, fold_idx=None):
        print(f"==> Building student model for fold {fold_idx + 1}..")
        try:
            if self.dataset_mode == 'hardfake':
                self.student = ResNet_50_sparse_hardfakevsreal()
            else:  # rvf10k
                self.student = ResNet_50_sparse_rvf10k()
            
            # مسیر چک‌پوینت برای فولد خاص
            ckpt_path = os.path.join(
                self.result_dir,
                f"student_model_fold_{fold_idx + 1}",
                f"finetune_{self.arch}_sparse_fold_{fold_idx + 1}_best.pt"
            )
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt_student = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            self.student.load_state_dict(state_dict, strict=True)
            self.student.to(self.device)
            print(f"Model for fold {fold_idx + 1} loaded on {self.device}")
            return ckpt_student.get("best_prec1_after_finetune", 0)  # دقت اعتبارسنجی
        except Exception as e:
            print(f"Error building model for fold {fold_idx + 1}: {str(e)}")
            raise

    def test(self, fold_idx):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.student.eval()
        self.student.ticket = True
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc=f"Test Fold {fold_idx + 1}") as _tqdm:
                    for images, targets in self.test_loader:
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

            print(f"[Test Fold {fold_idx + 1}] Dataset: {self.dataset_mode}, Prec@1: {meter_top1.avg:.2f}%")
            return meter_top1.avg
        except Exception as e:
            print(f"Error during testing fold {fold_idx + 1}: {str(e)}")
            raise

    def main(self):
        try:
            self.dataload()
            fold_results = []
            best_fold = None
            best_val_acc = 0

            # تست برای هر فولد
            for fold_idx in range(self.n_folds):
                print(f"\nTesting Fold {fold_idx + 1}/{self.n_folds}")
                val_acc = self.build_model(fold_idx)
                test_acc = self.test(fold_idx)
                fold_results.append({
                    'fold': fold_idx + 1,
                    'val_acc': val_acc,
                    'test_acc': test_acc
                })
                
                # به‌روزرسانی بهترین فولد
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_fold = fold_idx + 1

            # گزارش میانگین و انحراف معیار
            test_accs = [result['test_acc'] for result in fold_results]
            avg_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)
            print(f"\nK-Fold Test Results:")
            print(f"Average test accuracy across {self.n_folds} folds: {avg_test_acc:.2f} ± {std_test_acc:.2f}%")
            print(f"Best fold: Fold {best_fold} with validation accuracy: {best_val_acc:.2f}%")

            # محاسبه FLOPs و پارامترها برای بهترین مدل
            print(f"\nCalculating FLOPs and Params for best model (Fold {best_fold})")
            self.build_model(best_fold - 1)  # بارگذاری مدل بهترین فولد
            (
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
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise
