import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Subset

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}
        self.missing_images = 0  # برای شمارش تصاویر گمشده

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            self.missing_images += 1
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

    def get_missing_images_count(self):
        return self.missing_images

class DatasetSelector:
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
        k_folds=5,  # تعداد foldها برای K-Fold Cross Validation
        seed=3407,  # برای تکرارپذیری
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', or '140k'")

        self.dataset_mode = dataset_mode
        self.k_folds = k_folds
        self.seed = seed

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

        # Load full data based on dataset_mode
        if dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided")
            full_data = pd.read_csv(hardfake_csv_file)

            def create_image_path(row):
                folder = 'fake' if row['label'] in [0, 'fake', 'Fake'] else 'real'
                img_name = row['images_id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join(folder, img_name)

            full_data['images_id'] = full_data.apply(create_image_path, axis=1)
            root_dir = hardfake_root_dir
            test_data = None  # تست جداگانه ندارد

        elif dataset_mode == 'rvf10k':
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided")
            train_data = pd.read_csv(rvf10k_train_csv)
            valid_data = pd.read_csv(rvf10k_valid_csv)

            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'
                img_name = row['id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join('rvf10k', split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data['images_id'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)
            full_data = pd.concat([train_data, valid_data]).reset_index(drop=True)
            root_dir = rvf10k_root_dir
            test_data = None  # تست جداگانه ندارد

        else:  # dataset_mode == '140k'
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            valid_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            full_data = pd.concat([train_data, valid_data]).reset_index(drop=True)  # فقط train و valid برای K-Fold
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

            if 'path' not in full_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'path' column")

        # Create full dataset
        full_dataset = FaceDataset(full_data, root_dir, transform=transform_train, img_column=img_column)

        # Initialize K-Fold
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        # Store DataLoaders for each fold
        self.fold_loaders = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
            # Create Subset for train and validation
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # Create DataLoader for train
            if ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, shuffle=True)
                train_loader = DataLoader(
                    train_subset, batch_size=train_batch_size, num_workers=num_workers,
                    pin_memory=pin_memory, sampler=train_sampler,
                )
            else:
                train_loader = DataLoader(
                    train_subset, batch_size=train_batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory,
                )

            # Create DataLoader for validation
            val_loader = DataLoader(
                val_subset, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

            # محاسبه آمار توزیع برچسب‌ها
            train_labels = full_data.iloc[train_idx]['label'].map(self.label_map)
            val_labels = full_data.iloc[val_idx]['label'].map(self.label_map)
            train_real_count = sum(train_labels == 1)
            train_fake_count = sum(train_labels == 0)
            val_real_count = sum(val_labels == 1)
            val_fake_count = sum(val_labels == 0)

            self.fold_loaders.append({
                'fold': fold,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'train_stats': {'real': train_real_count, 'fake': train_fake_count},
                'val_stats': {'real': val_real_count, 'fake': val_fake_count}
            })

        # Create test DataLoader if available (only for 140k)
        if test_data is not None:
            test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)
            self.loader_test = DataLoader(
                test_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            self.loader_test = None

        # Debug: Print fold statistics
        print(f"{dataset_mode} dataset statistics for {self.k_folds}-Fold Cross Validation:")
        for fold_data in self.fold_loaders:
            fold = fold_data['fold']
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']
            train_stats = fold_data['train_stats']
            val_stats = fold_data['val_stats']
            print(f"Fold {fold + 1}:")
            print(f"  Train batches: {len(train_loader)}, Real: {train_stats['real']}, Fake: {train_stats['fake']}")
            print(f"  Validation batches: {len(val_loader)}, Real: {val_stats['real']}, Fake: {val_stats['fake']}")

        if self.loader_test:
            print(f"Test batches: {len(self.loader_test)}")
            try:
                sample = next(iter(self.loader_test))
                print(f"Sample test batch image shape: {sample[0].shape}")
                print(f"Sample test batch labels: {sample[1]}")
            except Exception as e:
                print(f"Error loading sample test batch: {e}")

        # گزارش تصاویر گمشده
        print(f"Total missing images: {full_dataset.get_missing_images_count()}")
        if self.loader_test:
            print(f"Test dataset missing images: {test_dataset.get_missing_images_count()}")

if __name__ == "__main__":
    # Example for hardfakevsrealfaces
    dataset_hardfake = DatasetSelector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=64,
        eval_batch_size=64,
        k_folds=5,
    )

    # Example for rvf10k
    dataset_rvf10k = DatasetSelector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=64,
        eval_batch_size=64,
        k_folds=5,
    )

    # Example for 140k Real and Fake Faces
    dataset_140k = DatasetSelector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=64,
        eval_batch_size=64,
        k_folds=5,
    )
