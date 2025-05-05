import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
import numpy as np

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
        n_folds=5,  # Number of folds for cross-validation
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', or '140k'")

        self.dataset_mode = dataset_mode
        self.n_folds = n_folds
        self.ddp = ddp

        # Define image size based on dataset_mode
        image_size = (256, 256) if dataset_mode in ['rvf10k', '140k'] else (300, 300)

        # Define transforms
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size[0], padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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

            # Split into train+val and test
            train_val_data, test_data = train_test_split(
                full_data, test_size=0.15, stratify=full_data['label'], random_state=3407
            )
            train_val_data = train_val_data.reset_index(drop=True)
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

            # Combine train and valid for cross-validation
            train_val_data = pd.concat([train_data, valid_data], ignore_index=True)
            train_val_data = train_val_data.sample(frac=1, random_state=3407).reset_index(drop=True)

            # Split off test set
            train_val_data, test_data = train_test_split(
                train_val_data, test_size=0.15, stratify=train_val_data['label'], random_state=3407
            )
            train_val_data = train_val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            root_dir = rvf10k_root_dir

        else:  # dataset_mode == '140k'
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            valid_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

            if 'path' not in train_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'path' column")

            # Combine train and valid for cross-validation
            train_val_data = pd.concat([train_data, valid_data], ignore_index=True)
            train_val_data = train_val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        # Debug: Print data statistics
        print(f"{dataset_mode} dataset statistics:")
        print(f"Total train+val dataset size: {len(train_val_data)}")
        print(f"Train+val label distribution:\n{train_val_data['label'].value<|control349|>}")
        print(f"Total test dataset size: {len(test_data)}")
        print(f"Test label distribution:\n{test_data['label'].value_counts()}")

        # Check for missing images
        for split, data in [('train+val', train_val_data), ('test', test_data)]:
            missing_images = []
            for img_path in data[img_column]:
                full_path = os.path.join(root_dir, img_path)
                if not os.path.exists(full_path):
                    missing_images.append(full_path)
            if missing_images:
                print(f"Missing {split} images: {len(missing_images)}")
                print(f"Sample missing {split} images:", missing_images[:5])

        # Create test dataset and loader
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)
        self.loader_test = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # Create base dataset for train+val
        train_val_dataset = FaceDataset(train_val_data, root_dir, transform=transform_train, img_column=img_column)

        # Initialize K-Fold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=3407)
        self.fold_loaders = []

        # Create DataLoader for each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_data)):
            print(f"\nFold {fold + 1} statistics:")
            train_data_fold = train_val_data.iloc[train_idx].reset_index(drop=True)
            val_data_fold = train_val_data.iloc[val_idx].reset_index(drop=True)
            print(f"Train fold size: {len(train_data_fold)}")
            print(f"Train fold label distribution:\n{train_data_fold['label'].value_counts()}")
            print(f"Validation fold size: {len(val_data_fold)}")
            print(f"Validation fold label distribution:\n{val_data_fold['label'].value_counts()}")

            # Create datasets for this fold
            train_dataset_fold = Subset(train_val_dataset, train_idx)
            val_dataset_fold = Subset(train_val_dataset, val_idx)

            # Create DataLoader for training
            if ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_fold, shuffle=True)
                train_loader = DataLoader(
                    train_dataset_fold, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
            else:
                train_loader = DataLoader(
                    train_dataset_fold, batch_size=train_batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory,
                )

            # Create DataLoader for validation
            val_loader = DataLoader(
                val_dataset_fold, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

            self.fold_loaders.append((train_loader, val_loader))

        # Debug: Print loader sizes
        for fold, (train_loader, val_loader) in enumerate(self.fold_loaders):
            print(f"Fold {fold + 1}:")
            print(f"  Train loader batches: {len(train_loader)}")
            print(f"  Validation loader batches: {len(val_loader)}")

        # Test a sample batch from test loader
        try:
            sample = next(iter(self.loader_test))
            print(f"Sample test batch image shape: {sample[0].shape}")
            print(f"Sample test batch labels: {sample[1]}")
        except Exception as e:
            print(f"Error loading sample test batch: {e}")

if __name__ == "__main__":
    # Example for hardfakevsrealfaces
    dataset_hardfake = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=64,
        eval_batch_size=64,
        n_folds=5,
    )

    # Example for rvf10k
    dataset_rvf10k = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=64,
        eval_batch_size=64,
        n_folds=5,
    )

    # Example for 140k Real and Fake Faces
    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=64,
        eval_batch_size=64,
        n_folds=5,
    )
