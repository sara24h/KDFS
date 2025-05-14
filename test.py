import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# فرض می‌کنیم این ماژول‌ها در پروژه شما وجود دارند
# اگر وجود ندارند، باید آنها را پیاده‌سازی کنید
from utils import meter
from get_flops_and_params import get_flops_and_params

# مدل‌های ResNet فرضی (باید با پیاده‌سازی واقعی جایگزین شوند)
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k, ResNet_50_sparse_140k

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            print(f"Warning: Image not found: {img_name}")
            image_size = 256 if '140k' in self.root_dir else 300
            image = Image.new('RGB', (image_size, image_size), color='black')
            label = self.label_map[self.data['label'].iloc[idx]]
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float)
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector(Dataset):
    def __init__(
        self,
        dataset_mode,  # 'hardfake', 'rvf10k', or '140k'
        hardfake_csv_file=None,
        hardfake_root_dir=None,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_root_dir=None,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', or '140k'")

        self.dataset_mode = dataset_mode

        # Define image size based on dataset_mode
        image_size = (256, 256) if dataset_mode in ['rvf10k', '140k'] else (300, 300)

        # Define mean and std based on dataset_mode
        if dataset_mode == 'hardfake':
            mean = (0.5124, 0.4165, 0.3684)
            std = (0.2363, 0.2087, 0.2029)
        elif dataset_mode == 'rvf10k':
            mean = (0.5212, 0.4260, 0.3811)
            std = (0.2486, 0.2238, 0.2211)
        else:  # dataset_mode == '140k'
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)

        # Define transforms
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size[0], padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Set img_column based on dataset_mode
        img_column = 'path' if dataset_mode == '140k' else 'images_id'

        # Load data based on dataset_mode
        if dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided")
            full_data = pd.read_csv(hardfake_csv_file)

            def create_image_path(row):
                folder = 'fake' if row['label'] == 'fake' else 'real'
                img_name = row['images_id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join(folder, img_name)

            full_data['images_id'] = full_data.apply(create_image_path, axis=1)
            root_dir = hardfake_root_dir

            train_data, temp_data = train_test_split(
                full_data, test_size=0.3, stratify=full_data['label'], random_state=3407
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=temp_data['label'], random_state=3407
            )
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

        elif dataset_mode == 'rvf10k':
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided")
            train_data = pd.read_csv(rvf10k_train_csv)

            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'
                img_name = row['id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join('rvf10k', split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data = pd.read_csv(rvf10k_valid_csv)
            valid_data['images_id'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)

            val_data, test_data = train_test_split(
                valid_data, test_size=0.5, stratify=valid_data['label'], random_state=3407
            )
            val_data = val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            root_dir = rvf10k_root_dir

        else:  # dataset_mode == '140k'
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            val_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

            if 'path' not in train_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'path' column")

            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        # Debug: Print data statistics
        print(f"{dataset_mode} dataset statistics:")
        print(f"Sample train image paths:\n{train_data[img_column].head()}")
        print(f"Total train dataset size: {len(train_data)}")
        print(f"Train label distribution:\n{train_data['label'].value_counts()}")
        print(f"Sample validation image paths:\n{val_data[img_column].head()}")
        print(f"Total validation dataset size: {len(val_data)}")
        print(f"Validation label distribution:\n{val_data['label'].value_counts()}")
        print(f"Sample test image paths:\n{test_data[img_column].head()}")
        print(f"Total test dataset size: {len(test_data)}")
        print(f"Test label distribution:\n{test_data['label'].value_counts()}")

        # Check for missing images
        for split, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
            missing_images = []
            for img_path in data[img_column]:
                full_path = os.path.join(root_dir, img_path)
                if not os.path.exists(full_path):
                    missing_images.append(full_path)
            if missing_images:
                print(f"Missing {split} images: {len(missing_images)}")
                print(f"Sample missing {split} images:", missing_images[:5])

        # Create datasets
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

        # Create data loaders
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            
            self.loader_train = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                num_workers=num_workers,
                pin_memory=pin_memory, 
                sampler=train_sampler,
            )
            self.loader_val = DataLoader(
                val_dataset, 
                batch_size=eval_batch_size, 
                num_workers=num_workers,
                pin_memory=pin_memory, 
                sampler=val_sampler,
            )
            self.loader_test = DataLoader(
                test_dataset, 
                batch_size=eval_batch_size, 
                num_workers=num_workers,
                pin_memory=pin_memory, 
                sampler=test_sampler,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset, 
                batch_size=train_batch_size, 
                shuffle=True,
                num_workers=num_workers, 
                pin_memory=pin_memory,
            )
            self.loader_val = DataLoader(
                val_dataset, 
                batch_size=eval_batch_size, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=pin_memory,
            )
            self.loader_test = DataLoader(
                test_dataset, 
                batch_size=eval_batch_size, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=pin_memory,
            )

        # Debug: Print loader sizes
        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_val)}")
        print(f"Test loader batches: {len(self.loader_test)}")

        # Test a sample batch
        for loader, name in [(self.loader_train, 'train'), (self.loader_val, 'validation'), (self.loader_test, 'test')]:
            try:
                sample = next(iter(loader))
                print(f"Sample {name} batch image shape: {sample[0].shape}")
                print(f"Sample {name} batch labels: {sample[1]}")
            except Exception as e:
                print(f"Error loading sample {name} batch: {e}")

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch  # Expected to be 'ResNet_50'
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode  # 'hardfake', 'rvf10k', or '140k'

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            # Verify dataset paths
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
            elif self.dataset_mode == 'rvf10k':
                train_csv = os.path.join(self.dataset_dir, 'train.csv')
                valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"CSV files not found: {train_csv}, {valid_csv}")
            else:  # 140k
                test_csv = os.path.join(self.dataset_dir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")

            # Initialize dataset based on mode
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
            else:  # 140k
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

            self.test_loader = dataset.loader_test
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self):
        print("==> Building student model..")
        try:
            print(f"Loading sparse student model for dataset mode: {self.dataset_mode}")
            if self.dataset_mode == 'hardfake':
                self.student = ResNet_50_sparse_hardfakevsreal()
            elif self.dataset_mode == 'rvf10k':
                self.student = ResNet_50_sparse_rvf10k()
            else:  # 140k
                self.student = ResNet_50_sparse_140k()

            # Load checkpoint
            if not os.path.exists(self.sparsed_student_ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            try:
                self.student.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"State dict loading failed with strict=True: {str(e)}")
                print("Trying with strict=False to identify mismatched keys...")
                self.student.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False; check for missing or unexpected keys.")

            self.student.to(self.device)
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self):
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.student.eval()
        self.student.ticket = True  # Enable ticket mode for sparse model
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc="Test") as _tqdm:
                    for images, targets in self.test_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze()
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100.0 * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)

                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            print(f"[Test] Dataset: {self.dataset_mode}, Prec@1: {meter_top1.avg:.2f}%")

            # Calculate FLOPs and parameters
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
            print(f"Error during testing: {str(e)}")
            raise

    def main(self):
        print(f"Starting test pipeline with dataset mode: {self.dataset_mode}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            self.dataload()
            self.build_model()
            self.test()
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise
