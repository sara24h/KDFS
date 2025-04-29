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
        self.label_map = {'real': 1, 'fake': 0}

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


class Dataset_hardfakevsreal(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        train_batch_size,
        eval_batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        # Define transforms
        transform_train = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(300, padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Load and preprocess the CSV file
        full_data = pd.read_csv(csv_file)

        # Modify images_id to include folder and .jpg extension
        def create_full_image_path(row):
            folder = 'fake' if row['label'] == 'fake' else 'real'
            img_name = row['images_id']
            # Remove any existing folder prefixes to avoid duplication
            img_name = os.path.basename(img_name)
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            return os.path.join(folder, img_name)

        full_data['images_id'] = full_data.apply(create_full_image_path, axis=1)

        # Debug: Print data statistics
        print("Sample image paths:")
        print(full_data['images_id'].head())
        print(f"Total dataset size: {len(full_data)}")
        print(f"Duplicate rows: {full_data.duplicated().sum()}")
        print(f"Label distribution:\n{full_data['label'].value_counts()}")
        print(f"Unique labels: {full_data['label'].unique()}")

        # Check for missing images
        missing_images = []
        for img_path in full_data['images_id']:
            full_path = os.path.join(root_dir, img_path)
            if not os.path.exists(full_path):
                missing_images.append(full_path)
        if missing_images:
            print(f"Missing images: {len(missing_images)}")
            print("Sample missing images:", missing_images[:5])

        # Shuffle and split the data with stratified sampling
        train_data, val_data = train_test_split(
            full_data,
            test_size=0.2,
            stratify=full_data['label'],
            random_state=3407
        )
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)

        # Debug: Print train and validation statistics
        print(f"Train dataset size: {len(train_data)}")
        print(f"Validation dataset size: {len(val_data)}")
        print(f"Train label distribution:\n{train_data['label'].value_counts()}")
        print(f"Validation label distribution:\n{val_data['label'].value_counts()}")

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
            print(f"Sample batch image shape: {sample[0].shape}")
            print(f"Sample batch labels: {sample[1]}")
        except Exception as e:
            print(f"Error loading sample batch: {e}")


if __name__ == "__main__":
    dataset = Dataset_hardfakevsreal(
        csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=32,
        eval_batch_size=32,
    )



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
        self.label_map = {'real': 1, 'fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data['images_id'].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {img_name}: {e}")
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class FakeVsReal10kDataset:
    def __init__(
        self,
        train_csv_file,
        valid_csv_file,
        root_dir,
        train_batch_size=32,
        valid_batch_size=32,
        test_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
        test_split_ratio=0.33,  # نسبت دیتاست تست از دیتاست اعتبارسنجی
    ):
        # تعریف تبدیلات
        transform_train = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(300, padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        transform_eval = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # بارگذاری فایل‌های CSV
        train_data = pd.read_csv(train_csv_file)
        valid_data = pd.read_csv(valid_csv_file)

        # تقسیم دیتاست اعتبارسنجی به اعتبارسنجی و تست
        valid_data, test_data = train_test_split(
            valid_data,
            test_size=test_split_ratio,
            stratify=valid_data['label'],
            random_state=3407
        )
        valid_data = valid_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # اصلاح مسیر تصاویر
        def create_full_image_path(row, folder_type):
            folder = 'fake' if row['label'] == 'fake' else 'real'
            img_name = row['images_id']
            img_name = os.path.basename(img_name)  # حذف پیشوندهای احتمالی
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            return os.path.join(folder_type, folder, img_name)

        train_data['images_id'] = train_data.apply(
            lambda row: create_full_image_path(row, 'train'), axis=1
        )
        valid_data['images_id'] = valid_data.apply(
            lambda row: create_full_image_path(row, 'valid'), axis=1
        )
        test_data['images_id'] = test_data.apply(
            lambda row: create_full_image_path(row, 'valid'), axis=1
        )  # تست از پوشه valid استفاده می‌کند

        # دیباگ: چاپ آمار دیتاست‌ها
        for name, data in [('Train', train_data), ('Validation', valid_data), ('Test', test_data)]:
            print(f"\n{name} dataset statistics:")
            print("Sample image paths:", data['images_id'].head().tolist())
            print(f"Total size: {len(data)}")
            print(f"Duplicate rows: {data.duplicated().sum()}")
            print(f"Label distribution:\n{data['label'].value_counts()}")
            print(f"Unique labels: {data['label'].unique()}")

            # بررسی تصاویر گم‌شده
            missing_images = []
            for img_path in data['images_id']:
                full_path = os.path.join(root_dir, img_path)
                if not os.path.exists(full_path):
                    missing_images.append(full_path)
            if missing_images:
                print(f"Missing images: {len(missing_images)}")
                print("Sample missing images:", missing_images[:5])
            else:
                print("No missing images.")

        # ایجاد دیتاست‌ها
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        valid_dataset = FaceDataset(valid_data, root_dir, transform=transform_eval)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_eval)

        # ایجاد دیتالودرها
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

        self.loader_valid = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.loader_test = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # دیباگ: چاپ اطلاعات دیتالودرها
        print(f"\nTrain loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_valid)}")
        print(f"Test loader batches: {len(self.loader_test)}")
        try:
            sample = next(iter(self.loader_train))
            print(f"Sample batch image shape: {sample[0].shape}")
            print(f"Sample batch labels: {sample[1]}")
        except Exception as e:
            print(f"Error loading sample batch: {e}")


if __name__ == "__main__":
    dataset = FakeVsReal10kDataset(
        train_csv_file='/kaggle/input/rvf10k/train.csv',
        valid_csv_file='/kaggle/input/rvf10k/valid.csv',
        root_dir='/kaggle/input/rvf10k',
        train_batch_size=32,
        valid_batch_size=32,
        test_batch_size=32,
        test_split_ratio=0.33,  # 33% از دیتاست اعتبارسنجی برای تست
    )
