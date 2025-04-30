import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        # اصلاح label_map برای سازگاری با مقادیر عددی 1 و 0
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0}  # پشتیبانی از هر دو فرمت

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data['images_id'].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class Dataset_selector(Dataset):
    def __init__(
        self,
        dataset_mode,  # 'hardfake' or 'rvf10k'
        hardfake_csv_file=None,
        hardfake_root_dir=None,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode not in ['hardfake', 'rvf10k']:
            raise ValueError("dataset_mode must be 'hardfake' or 'rvf10k'")

        self.dataset_mode = dataset_mode

        # Define image size based on dataset_mode
        image_size = (256, 256) if dataset_mode == 'rvf10k' else (300, 300)

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

        # Load data based on dataset_mode
        if dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided for hardfake mode")
            # Load and preprocess hardfakevsrealfaces
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

            # Split into train and validation
            train_data, val_data = train_test_split(
                full_data,
                test_size=0.2,
                stratify=full_data['label'],
                random_state=3407
            )
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)

        else:  # dataset_mode == 'rvf10k'
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided for rvf10k mode")
            # Load rvf10k train data
            train_data = pd.read_csv(rvf10k_train_csv)

            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'  # اصلاح برای مقادیر عددی
                img_name = row['id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join('rvf10k', split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)

            # Load rvf10k valid data
            val_data = pd.read_csv(rvf10k_valid_csv)
            val_data['images_id'] = val_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)
            root_dir = rvf10k_root_dir

        # Debug: Print data statistics
        print(f"{dataset_mode} dataset statistics:")
        print(f"Sample train image paths:\n{train_data['images_id'].head()}")
        print(f"Total train dataset size: {len(train_data)}")
        print(f"Train label distribution:\n{train_data['label'].value_counts()}")
        print(f"Sample validation image paths:\n{val_data['images_id'].head()}")
        print(f"Total validation dataset size: {len(val_data)}")
        print(f"Validation label distribution:\n{val_data['label'].value_counts()}")

        # Check for missing images
        missing_train_images = []
        for img_path in train_data['images_id']:
            full_path = os.path.join(root_dir, img_path)
            if not os.path.exists(full_path):
                missing_train_images.append(full_path)
        if missing_train_images:
            print(f"Missing train images: {len(missing_train_images)}")
            print("Sample missing train images:", missing_train_images[:5])

        missing_val_images = []
        for img_path in val_data['images_id']:
            full_path = os.path.join(root_dir, img_path)
            if not os.path.exists(full_path):
                missing_val_images.append(full_path)
        if missing_val_images:
            print(f"Missing validation images: {len(missing_val_images)}")
            print("Sample missing validation images:", missing_val_images[:5])

        # Create datasets
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test)

        # Create data loaders
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
            self.loader_train = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        self.loader_test = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Debug: Print loader sizes
        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_test)}")

        # Test a sample batch
        try:
            sample = next(iter(self.loader_train))
            print(f"Sample train batch image shape: {sample[0].shape}")
            print(f"Sample train batch labels: {sample[1]}")
        except Exception as e:
            print(f"Error loading sample train batch: {e}")

        try:
            sample = next(iter(self.loader_test))
            print(f"Sample validation batch image shape: {sample[0].shape}")
            print(f"Sample validation batch labels: {sample[1]}")
        except Exception as e:
            print(f"Error loading sample validation batch: {e}")


if __name__ == "__main__":
    # Example for hardfakevsrealfaces
    dataset_hardfake = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=32,
        eval_batch_size=32,
    )

    # Example for rvf10k
    dataset_rvf10k = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=32,
        eval_batch_size=32,
    )
