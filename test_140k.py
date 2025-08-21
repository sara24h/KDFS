import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import random
import time
from datetime import datetime
import json
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.seed = args.seed  # Add seed if not present
        self.device = args.device
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.result_dir = args.result_dir
        self.new_dataset_dir = getattr(args, 'new_dataset_dir', None)
        self.f_lr = args.f_lr  # Fine-tune LR
        self.f_epochs = args.f_epochs  # Fine-tune epochs
        self.resume = getattr(args, 'resume', None)  # Optional resume

        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

        self.start_epoch = 0
        self.best_val_acc = 0.0

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "test_finetune_logger.log"), "test_finetune_logger"
            )
            self.logger.info("test and finetune config:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "config.txt")
            )
            self.logger.info("--------- Test and Finetune -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        self.seed = self.seed + self.rank
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
        if self.rank == 0:
            print("==> Loading datasets...")
        
        image_size = (256, 256)
        mean_140k = [0.5207, 0.4258, 0.3806]
        std_140k = [0.2490, 0.2239, 0.2212]

        transform_train_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        transform_val_test_140k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_140k, std=std_140k),
        ])

        params = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': True  # Enable DDP
        }
        
        if self.dataset_mode == 'hardfake':
            params['hardfake_csv_file'] = os.path.join(self.dataset_dir, 'data.csv')
            params['hardfake_root_dir'] = self.dataset_dir
        elif self.dataset_mode == 'rvf10k':
            params['rvf10k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['rvf10k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['rvf10k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '140k':
            params['realfake140k_train_csv'] = os.path.join(self.dataset_dir, 'train.csv')
            params['realfake140k_valid_csv'] = os.path.join(self.dataset_dir, 'valid.csv')
            params['realfake140k_test_csv'] = os.path.join(self.dataset_dir, 'test.csv')
            params['realfake140k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '200k':
            image_root_dir = os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset', 'my_real_vs_ai_dataset')
            params['realfake200k_root_dir'] = image_root_dir
            params['realfake200k_train_csv'] = os.path.join(self.dataset_dir, 'train_labels.csv')
            params['realfake200k_val_csv'] = os.path.join(self.dataset_dir, 'val_labels.csv')
            params['realfake200k_test_csv'] = os.path.join(self.dataset_dir, 'test_labels.csv')
        elif self.dataset_mode == '190k':
            params['realfake190k_root_dir'] = self.dataset_dir
        elif self.dataset_mode == '330k':
            params['realfake330k_root_dir'] = self.dataset_dir

        dataset_manager = Dataset_selector(**params)

        if self.rank == 0:
            print("Overriding transforms to use consistent 140k normalization stats for all datasets.")
        dataset_manager.loader_train.dataset.transform = transform_train_140k
        dataset_manager.loader_val.dataset.transform = transform_val_test_140k
        dataset_manager.loader_test.dataset.transform = transform_val_test_140k

        self.train_loader = dataset_manager.loader_train
        self.val_loader = dataset_manager.loader_val
        self.test_loader = dataset_manager.loader_test
        
        if self.rank == 0:
            print(f"All loaders for '{self.dataset_mode}' are now configured with 140k normalization.")

        # Load new test dataset if provided
        if self.new_dataset_dir:
            if self.rank == 0:
                print("==> Loading new test dataset...")
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir,
                'ddp': True
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_dataset_manager.loader_test.dataset.transform = transform_val_test_140k
            self.new_test_loader = new_dataset_manager.loader_test
            if self.rank == 0:
                print(f"New test dataset loader configured with 140k normalization.")

    def build_model(self):
        if self.rank == 0:
            print("==> Building student model...")
        self.student = ResNet_50_sparse_hardfakevsreal()
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            
        if self.rank == 0:
            print(f"Loading pre-trained weights from: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt_student.get("student", ckpt_student)
        
        # Handle potential 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.student.load_state_dict(state_dict, strict=False)
        self.student = self.student.cuda()
        self.student = DDP(self.student, device_ids=[self.local_rank])
        if self.rank == 0:
            print(f"Model loaded on {self.device}")

    def compute_metrics(self, loader, description="Test", print_metrics=True, save_confusion_matrix=True):
        self.student.eval()
        self.student.module.ticket = True
        all_preds = []
        all_targets = []
        sample_info = []
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100, disable=self.rank != 0)):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True).float()
                
                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                # Gather across processes
                gathered_preds = [torch.zeros_like(preds) for _ in range(self.world_size)]
                dist.all_gather(gathered_preds, preds)
                gathered_targets = [torch.zeros_like(targets) for _ in range(self.world_size)]
                dist.all_gather(gathered_targets, targets)
                
                all_preds.extend(torch.cat(gathered_preds).cpu().numpy())
                all_targets.extend(torch.cat(gathered_targets).cpu().numpy())
                
                # For sample_info, gather paths if available (simplified, assume no paths for distributed)
                # Note: Gathering full paths might be memory-intensive; skip or adapt as needed
                
                correct = (preds == targets).sum().item()
                reduced_correct = self.reduce_tensor(torch.tensor(correct).cuda()).item()
                reduced_batch_size = self.reduce_tensor(torch.tensor(images.size(0)).cuda()).item()
                prec1 = 100.0 * reduced_correct / reduced_batch_size if reduced_batch_size > 0 else 0
                meter_top1.update(prec1, reduced_batch_size)
        
        # Gather all_preds and all_targets to rank 0 for metrics computation
        all_preds_tensor = torch.tensor(all_preds).cuda()
        all_targets_tensor = torch.tensor(all_targets).cuda()
        dist.barrier()
        if self.rank == 0:
            gathered_preds_list = [torch.zeros_like(all_preds_tensor) for _ in range(self.world_size)]
            gathered_targets_list = [torch.zeros_like(all_targets_tensor) for _ in range(self.world_size)]
            dist.gather(all_preds_tensor, gather_list=gathered_preds_list)
            dist.gather(all_targets_tensor, gather_list=gathered_targets_list)
            all_preds = torch.cat(gathered_preds_list).cpu().numpy()
            all_targets = torch.cat(gathered_targets_list).cpu().numpy()
        else:
            dist.gather(all_preds_tensor)
            dist.gather(all_targets_tensor)
            all_preds = np.array([])
            all_targets = np.array([])
        
        if self.rank == 0:
            accuracy = meter_top1.avg
            precision = precision_score(all_targets, all_preds, average='binary')
            recall = recall_score(all_targets, all_preds, average='binary')
            
            precision_per_class = precision_score(all_targets, all_preds, average=None, labels=[0, 1])
            recall_per_class = recall_score(all_targets, all_preds, average=None, labels=[0, 1])
            
            tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
            specificity_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if print_metrics:
                print(f"[{description}] Overall Metrics:")
                print(f"Accuracy: {accuracy:.2f}%")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"Specificity: {specificity_real:.4f}")
                
                print(f"\n[{description}] Per-Class Metrics:")
                print(f"Class Real (0):")
                print(f"  Precision: {precision_per_class[0]:.4f}")
                print(f"  Recall: {recall_per_class[0]:.4f}")
                print(f"  Specificity: {specificity_real:.4f}")
                print(f"Class Fake (1):")
                print(f"  Precision: {precision_per_class[1]:.4f}")
                print(f"  Recall: {recall_per_class[1]:.4f}")
                print(f"  Specificity: {specificity_fake:.4f}")
            
            cm = confusion_matrix(all_targets, all_preds)
            classes = ['Real', 'Fake']
            
            if save_confusion_matrix:
                print(f"\n[{description}] Confusion Matrix:")
                print(f"{'':>10} {'Predicted Real':>15} {'Predicted Fake':>15}")
                print(f"{'Actual Real':>10} {cm[0,0]:>15} {cm[0,1]:>15}")
                print(f"{'Actual Fake':>10} {cm[1,0]:>15} {cm[1,1]:>15}")
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
                plt.title(f'Confusion Matrix - {description}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                sanitized_description = description.lower().replace(" ", "_").replace("/", "_")
                plot_path = os.path.join(self.result_dir, f'confusion_matrix_{sanitized_description}.png')
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)
                plt.close()
                print(f"Confusion matrix saved to: {plot_path}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity_real,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'specificity_per_class': [specificity_real, specificity_fake],
                'confusion_matrix': cm,
                'sample_info': []  # Simplified, no sample_info in distributed
            }
        else:
            return {}

    def display_samples(self, sample_info, description="Test", num_samples=30):
        if self.rank == 0:
            print(f"\n[{description}] Displaying first {num_samples} test samples:")
            print(f"{'Sample ID':<50} {'True Label':<12} {'Predicted Label':<12}")
            print("-" * 80)
            for i, sample in enumerate(sample_info[:num_samples]):
                true_label = 'Real' if sample['true_label'] == 0 else 'Fake'
                pred_label = 'Real' if sample['pred_label'] == 0 else 'Fake'
                print(f"{sample['id']:<50} {true_label:<12} {pred_label:<12}")

    def resume_ckpt(self):
        if self.resume and os.path.exists(self.resume):
            ckpt = torch.load(self.resume, map_location="cpu")
            self.best_val_acc = ckpt.get("best_val_acc", 0.0)
            self.start_epoch = ckpt.get("start_epoch", 0)
            self.student.module.load_state_dict(ckpt["student"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            if self.rank == 0:
                self.logger.info(f"Resumed from epoch {self.start_epoch}")
        else:
            if self.rank == 0:
                self.logger.info("No resume checkpoint found, starting from scratch.")

    def save_ckpt(self, is_best, epoch):
        if self.rank == 0:
            ckpt = {
                "best_val_acc": self.best_val_acc,
                "start_epoch": epoch,
                "student": self.student.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            folder = os.path.join(self.result_dir, "finetuned_model")
            os.makedirs(folder, exist_ok=True)
            if is_best:
                torch.save(ckpt, os.path.join(folder, f'finetuned_model_best_{self.dataset_mode}.pth'))
            torch.save(ckpt, os.path.join(folder, f'finetuned_model_last_{self.dataset_mode}.pth'))

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def finetune(self):
        if self.rank == 0:
            print("==> Fine-tuning using FEATURE EXTRACTOR strategy on 'fc' and 'layer4'...")
        
        for name, param in self.student.module.named_parameters():
            if 'fc' in name or 'layer4' in name:
                param.requires_grad = True
                if self.rank == 0:
                    print(f"Unfreezing for training: {name}")
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.module.parameters()),
            lr=self.f_lr,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.criterion = torch.nn.BCEWithLogitsLoss().cuda()
        
        self.student.module.ticket = False
        scaler = GradScaler()
        
        self.resume_ckpt()
        
        best_model_path = os.path.join(self.result_dir, f'finetuned_model_best_{self.dataset_mode}.pth')

        for epoch in range(self.start_epoch, self.f_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.train()
            meter_loss = meter.AverageMeter("Loss", ":6.4f")
            meter_top1_train = meter.AverageMeter("Train Acc@1", ":6.2f")
            
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.f_epochs} [Train]", ncols=100, disable=self.rank != 0) as _tqdm:
                for images, targets in self.train_loader:
                    images, targets = images.cuda(), targets.cuda().float()
                    self.optimizer.zero_grad()
                    
                    with autocast():
                        logits, _ = self.student(images)
                        logits = logits.squeeze()
                        loss = self.criterion(logits, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100.0 * correct / images.size(0)
                    
                    reduced_loss = self.reduce_tensor(loss)
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1).cuda())
                    reduced_n = self.reduce_tensor(torch.tensor(images.size(0)).cuda())
                    
                    meter_loss.update(reduced_loss.item(), reduced_n.item())
                    meter_top1_train.update(reduced_prec1.item(), reduced_n.item())
                    
                    _tqdm.update(1)
                    time.sleep(0.01)

            # Compute validation metrics
            val_metrics = self.compute_metrics(self.val_loader, description=f"Epoch_{epoch+1}_{self.f_epochs}_Val", print_metrics=False, save_confusion_matrix=False)
            val_acc = val_metrics.get('accuracy', 0.0) if self.rank == 0 else 0.0
            dist.broadcast(torch.tensor(val_acc).cuda(), src=0)
            val_acc = torch.tensor(val_acc).item()
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}: Train Loss: {meter_loss.avg:.4f}, Train Acc: {meter_top1_train.avg:.2f}%, Val Acc: {val_acc:.2f}%")

            self.scheduler.step()

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.rank == 0:
                    print(f"New best model found with Val Acc: {self.best_val_acc:.2f}%. Saving to {best_model_path}")
                self.save_ckpt(True, epoch + 1)
            self.save_ckpt(False, epoch + 1)
        
        if self.rank == 0:
            print(f"\nFine-tuning finished. Loading best model with Val Acc: {self.best_val_acc:.2f}%")
        self.resume_ckpt()  # Load best implicitly by resuming from last best
        
        # Compute and print final test metrics after fine-tuning
        final_test_metrics = self.compute_metrics(self.test_loader, description="Final_Test", print_metrics=True, save_confusion_matrix=True)

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        
        if self.rank == 0:
            print(f"Starting pipeline with dataset mode: {self.dataset_mode}")
            print("\n--- Testing BEFORE fine-tuning ---")
        initial_metrics = self.compute_metrics(self.test_loader, "Initial_Test")
        self.display_samples(initial_metrics.get('sample_info', []), "Initial Test", num_samples=30)
        
        if self.rank == 0:
            print("\n--- Starting fine-tuning ---")
        self.finetune()
        
        if self.rank == 0:
            print("\n--- Testing AFTER fine-tuning with best model ---")
        final_metrics = self.compute_metrics(self.test_loader, "Final_Test", print_metrics=False)
        self.display_samples(final_metrics.get('sample_info', []), "Final Test", num_samples=30)
        
        if self.new_test_loader:
            if self.rank == 0:
                print("\n--- Testing on NEW dataset ---")
            new_metrics = self.compute_metrics(self.new_test_loader, "New_Dataset_Test")
            self.display_samples(new_metrics.get('sample_info', []), "New Dataset Test", num_samples=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune model for real vs fake image detection')
    
    # Dataset selection
    parser.add_argument('--dataset_mode', type=str, default='140k', 
                        choices=['hardfake', 'rvf10k', '140k', '200k', '190k', '330k'],
                        help='Dataset mode to use (e.g., 140k, 200k)')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to the dataset directory')
    
    # Model selection
    parser.add_argument('--arch', type=str, default='resnet50_sparse', 
                        choices=['resnet50_sparse'], 
                        help='Model architecture to use')
    parser.add_argument('--sparsed_student_ckpt_path', type=str, required=True, 
                        help='Path to the pre-trained sparse student checkpoint')
    
    # Other required arguments
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', 
                        help='Pin memory for data loaders')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help='Batch size for testing')
    parser.add_argument('--result_dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--new_dataset_dir', type=str, default=None, 
                        help='Optional new dataset directory for additional testing')
    parser.add_argument('--f_lr', type=float, default=0.001, 
                        help='Learning rate for fine-tuning')
    parser.add_argument('--f_epochs', type=int, default=10, 
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Set if needed
    
    test = Test(args)
    test.main()
