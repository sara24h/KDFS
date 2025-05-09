import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# کلاس‌های FaceDataset و Dataset_selector (از کد خودتون)
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
        dataset_mode,
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
        image_size = (256, 256) if dataset_mode in ['rvf10k', '140k'] else (300, 300)

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

        img_column = 'path' if dataset_mode == '140k' else 'images_id'

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

        else:
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

        for split, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
            missing_images = []
            for img_path in data[img_column]:
                full_path = os.path.join(root_dir, img_path)
                if not os.path.exists(full_path):
                    missing_images.append(full_path)
            if missing_images:
                print(f"Missing {split} images: {len(missing_images)}")
                print(f"Sample missing {split} images:", missing_images[:5])

        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.loader_train = DataLoader(
                train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                pin_memory=pin_memory, sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.loader_val = DataLoader(
            val_dataset, batch_size=eval_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.loader_test = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_val)}")
        print(f"Test loader batches: {len(self.loader_test)}")

        for loader, name in [(self.loader_train, 'train'), (self.loader_val, 'validation'), (self.loader_test, 'test')]:
            try:
                sample = next(iter(loader))
                print(f"Sample {name} batch image shape: {sample[0].shape}")
                print(f"Sample {name} batch labels: {sample[1]}")
            except Exception as e:
                print(f"Error loading sample {name} batch: {e}")

# تابع محاسبه mean و std
def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std

if __name__ == "__main__":
    # ساخت نمونه‌های Dataset_selector
    dataset_hardfake = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=64,
        eval_batch_size=64,
    )

    dataset_rvf10k = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=64,
        eval_batch_size=64,
    )

    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=64,
        eval_batch_size=64,
    )

    # محاسبه برای hardfake
    image_size = (300, 300)
    transform_for_stats = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    train_dataset_for_stats = FaceDataset(
        data_frame=dataset_hardfake.loader_train.dataset.data,
        root_dir=dataset_hardfake.loader_train.dataset.root_dir,
        transform=transform_for_stats,
        img_column=dataset_hardfake.loader_train.dataset.img_column
    )
    loader_for_stats = DataLoader(
        train_dataset_for_stats, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )
    print(f"Number of training samples for hardfake: {len(train_dataset_for_stats)}")
    mean_hardfake, std_hardfake = compute_mean_std(loader_for_stats)
    print(f"Hardfake dataset: mean={mean_hardfake}, std={std_hardfake}")

    # محاسبه برای rvf10k
    image_size = (256, 256)
    transform_for_stats = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    train_dataset_for_stats = FaceDataset(
        data_frame=dataset_rvf10k.loader_train.dataset.data,
        root_dir=dataset_rvf10k.loader_train.dataset.root_dir,
        transform=transform_for_stats,
        img_column=dataset_rvf10k.loader_train.dataset.img_column
    )
    loader_for_stats = DataLoader(
        train_dataset_for_stats, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )
    print(f"Number of training samples for rvf10k: {len(train_dataset_for_stats)}")
    mean_rvf10k, std_rvf10k = compute_mean_std(loader_for_stats)
    print(f"RVF10k dataset: mean={mean_rvf10k}, std={std_rvf10k}")

    # محاسبه برای 140k
    image_size = (256, 256)
    transform_for_stats = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    train_dataset_for_stats = FaceDataset(
        data_frame=dataset_140k.loader_train.dataset.data,
        root_dir=dataset_140k.loader_train.dataset.root_dir,
        transform=transform_for_stats,
        img_column=dataset_140k.loader_train.dataset.img_column
    )
    loader_for_stats = DataLoader(
        train_dataset_for_stats, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )
    print(f"Number of training samples for 140k: {len(train_dataset_for_stats)}")
    mean_140k, std_140k = compute_mean_std(loader_for_stats)
    print(f"140k dataset: mean={mean_140k}, std={std_140k}")
