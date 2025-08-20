import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter
from torch.cuda.amp import autocast, GradScaler
import random
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Test and Fine-tune Sparse ResNet Model with DDP")
    
    # آرگومان‌های برای فاین‌تیون
    parser.add_argument('--finetune_dataset_mode', type=str, required=True,
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help="Dataset mode for fine-tuning (train/val)")
    parser.add_argument('--finetune_dataset_dir', type=str, required=True,
                        help="Directory path to the fine-tuning dataset")
    
    # آرگومان‌های برای تست اصلی
    parser.add_argument('--test_dataset_mode', type=str, required=True,
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help="Dataset mode for main test")
    parser.add_argument('--test_dataset_dir', type=str, required=True,
                        help="Directory path to the main test dataset")
    
    # آرگومان‌های اختیاری برای تست اضافی
    parser.add_argument('--new_test_dataset_mode', type=str, default=None,
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k', None],
                        help="Dataset mode for additional new test (optional)")
    parser.add_argument('--new_test_dataset_dir', type=str, default=None,
                        help="Directory for additional new test dataset (optional)")
    
    # سایر آرگومان‌ها
    parser.add_argument('--sparsed_student_ckpt_path', type=str, required=True,
                        help="Path to the sparse student checkpoint")
    parser.add_argument('--result_dir', type=str, required=True,
                        help="Directory to save results")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help="Pin memory for data loaders")
    parser.add_argument('--arch', type=str, default='resnet50',
                        help="Model architecture")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help="Train batch size per GPU")
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help="Test/eval batch size per GPU")
    parser.add_argument('--f_lr', type=float, default=0.001,
                        help="Fine-tuning learning rate")
    parser.add_argument('--f_epochs', type=int, default=10,
                        help="Number of fine-tuning epochs")
    
    args = parser.parse_args()
    return args

class TestDDP:
    def __init__(self, args):
        self.args = args
        self.dist_init()
        self.setup_seed()
        self.result_init()

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None
        self.best_val_acc = 0.0

    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.args.result_dir):
                os.makedirs(self.args.result_dir)
            logging.basicConfig(filename=os.path.join(self.args.result_dir, "test_logger.log"), 
                                level=logging.INFO, format='%(asctime)s - %(message)s')
            self.logger = logging.getLogger("test_logger")
            print(f"Results will be saved to: {self.args.result_dir}")

    def setup_seed(self):
        seed = self.args.seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def dataload(self):
        if self.rank == 0:
            print("==> Loading datasets...")
        
        image_size = (256, 256)

        # Helper function to create the correct keyword arguments for Dataset_selector
        def get_dataset_kwargs(mode, path):
            kwargs = {}
            if mode == 'hardfake':
                kwargs['hardfake_root_dir'] = path
                kwargs['hardfake_csv_file'] = os.path.join(path, 'data.csv')
            elif mode == 'rvf10k':
                kwargs['rvf10k_root_dir'] = path
                kwargs['rvf10k_train_csv'] = os.path.join(path, 'train.csv')
                kwargs['rvf10k_valid_csv'] = os.path.join(path, 'valid.csv')
            elif mode == '140k':
                kwargs['realfake140k_root_dir'] = path
                kwargs['realfake140k_train_csv'] = os.path.join(path, 'train.csv')
                kwargs['realfake140k_valid_csv'] = os.path.join(path, 'valid.csv')
                kwargs['realfake140k_test_csv'] = os.path.join(path, 'test.csv')
            elif mode == '190k':
                kwargs['realfake190k_root_dir'] = path
            elif mode == '200k':
                kwargs['realfake200k_root_dir'] = path
                kwargs['realfake200k_train_csv'] = os.path.join(path, 'train_labels.csv')
                kwargs['realfake200k_val_csv'] = os.path.join(path, 'val_labels.csv')
                kwargs['realfake200k_test_csv'] = os.path.join(path, 'test_labels.csv')
            elif mode == '330k':
                kwargs['realfake330k_root_dir'] = path
            return kwargs

        # --- Create Fine-tuning Loader ---
        finetune_kwargs = get_dataset_kwargs(self.args.finetune_dataset_mode, self.args.finetune_dataset_dir)
        finetune_dataset_manager = Dataset_selector(
            dataset_mode=self.args.finetune_dataset_mode,
            train_batch_size=self.args.train_batch_size,
            eval_batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            ddp=True,
            **finetune_kwargs
        )
        
        # Read dynamic mean and std
        try:
            mean_dynamic = finetune_dataset_manager.mean
            std_dynamic = finetune_dataset_manager.std
            if self.rank == 0:
                print(f"==> Using dynamic normalization stats for '{self.args.finetune_dataset_mode}':")
                print(f"  Mean: {mean_dynamic}")
                print(f"  Std: {std_dynamic}")
        except AttributeError:
            mean_dynamic = [0.485, 0.456, 0.406]
            std_dynamic = [0.229, 0.224, 0.225]
            if self.rank == 0:
                print("==> Warning: Using default ImageNet normalization stats.")

        # Create transforms
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_dynamic, std=std_dynamic),
        ])
        transform_val_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_dynamic, std=std_dynamic),
        ])

        # Apply transforms to fine-tuning loaders
        self.train_loader = finetune_dataset_manager.loader_train
        self.val_loader = finetune_dataset_manager.loader_val
        self.train_loader.dataset.transform = transform_train
        self.val_loader.dataset.transform = transform_val_test
        
        # --- Create Main Test Loader ---
        test_kwargs = get_dataset_kwargs(self.args.test_dataset_mode, self.args.test_dataset_dir)
        test_dataset_manager = Dataset_selector(
            dataset_mode=self.args.test_dataset_mode,
            eval_batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            ddp=True,
            **test_kwargs
        )
        self.test_loader = test_dataset_manager.loader_test
        self.test_loader.dataset.transform = transform_val_test

        # --- Create Additional Test Loader (if provided) ---
        if self.args.new_test_dataset_dir and self.args.new_test_dataset_mode:
            new_test_kwargs = get_dataset_kwargs(self.args.new_test_dataset_mode, self.args.new_test_dataset_dir)
            new_test_manager = Dataset_selector(
                dataset_mode=self.args.new_test_dataset_mode,
                eval_batch_size=self.args.test_batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                ddp=True,
                **new_test_kwargs
            )
            self.new_test_loader = new_test_manager.loader_test
            self.new_test_loader.dataset.transform = transform_val_test

    def build_model(self):
        if self.rank == 0:
            print(f"==> Building model '{self.args.arch}'...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if self.rank == 0:
            print(f"==> Loading sparse student checkpoint from: {self.args.sparsed_student_ckpt_path}")
        
        ckpt = torch.load(self.args.sparsed_student_ckpt_path, map_location=self.device)
        self.student.load_state_dict(ckpt['model'], strict=False)
        self.student.to(self.device)
        
        self.student = DDP(self.student, device_ids=[self.local_rank])

    def finetune(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.args.f_lr)
        scaler = GradScaler()
        
        for epoch in range(self.args.f_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            
            # --- Training Phase ---
            self.student.train()
            train_loss = 0.0
            pbar = tqdm(self.train_loader, disable=(self.rank != 0))
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = self.student(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{self.args.f_epochs} | Train Loss: {loss.item():.4f}")

            # --- Validation Phase ---
            val_acc = self._evaluate(self.val_loader, "Validation")
            
            if self.rank == 0:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    save_path = os.path.join(self.args.result_dir, 'finetuned_best.pt')
                    torch.save(self.student.module.state_dict(), save_path)
                    print(f"Epoch {epoch+1}: New best model saved with accuracy: {val_acc:.4f}")
                    self.logger.info(f"Epoch {epoch+1}: New best model saved with accuracy: {val_acc:.4f}")
            
            dist.barrier()

    def test(self):
        if self.rank == 0:
            print("==> Loading best fine-tuned model for final test...")
        
        best_model_path = os.path.join(self.args.result_dir, 'finetuned_best.pt')
        if os.path.exists(best_model_path):
            self.student.module.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            if self.rank == 0:
                print("Warning: No best model found. Testing with the last model.")
        
        self._evaluate(self.test_loader, f"Main Test ({self.args.test_dataset_mode})", final_test=True)
        
        if self.new_test_loader:
            self._evaluate(self.new_test_loader, f"Additional Test ({self.args.new_test_dataset_mode})", final_test=True)

    def _evaluate(self, dataloader, phase_name, final_test=False):
        self.student.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(dataloader, disable=(self.rank != 0))
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds)
                all_labels.append(labels)
                pbar.set_description(f"Evaluating on {phase_name}")

        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        
        # جمع‌آوری نتایج از تمام پردازنده‌ها
        gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_preds, preds_tensor)
        dist.all_gather(gathered_labels, labels_tensor)

        if self.rank == 0:
            final_preds = torch.cat(gathered_preds).cpu().numpy()
            final_labels = torch.cat(gathered_labels).cpu().numpy()
            
            accuracy = accuracy_score(final_labels, final_preds)
            precision = precision_score(final_labels, final_preds, average='binary', zero_division=0)
            recall = recall_score(final_labels, final_preds, average='binary', zero_division=0)
            f1 = f1_score(final_labels, final_preds, average='binary', zero_division=0)

            print(f"\nResults for {phase_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            self.logger.info(f"Results for {phase_name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            if final_test:
                cm = confusion_matrix(final_labels, final_preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - {phase_name}')
                cm_path = os.path.join(self.args.result_dir, f'confusion_matrix_{phase_name.replace(" ", "_")}.png')
                plt.savefig(cm_path)
                plt.close()
                print(f"Confusion matrix saved to {cm_path}")

            return accuracy
        return 0.0

    def main(self):
        self.dataload()
        self.build_model()
        
        if self.rank == 0:
            print("\n" + "="*50)
            print("               STARTING FINE-TUNING")
            print("="*50 + "\n")
        self.finetune()
        
        dist.barrier()
        
        if self.rank == 0:
            print("\n" + "="*50)
            print("                STARTING FINAL TESTING")
            print("="*50 + "\n")
        self.test()
        
        if self.rank == 0:
            print("\nProcess finished successfully.")
        
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    test_ddp = TestDDP(args)
    test_ddp.main()
