import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image

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
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {img_name}")
            raise
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class HardFakeVsRealDataLoader:
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
        # تعریف تبدیل‌ها
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])

        # خواندن داده‌ها
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        full_data = pd.read_csv(csv_file)

        # بررسی وجود ستون‌های مورد نیاز
        required_columns = ['images_id', 'label']
        if not all(col in full_data.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # تصادفی کردن داده‌ها
        full_data = full_data.sample(frac=1).reset_index(drop=True)

        # تقسیم داده‌ها به آموزشی و اعتبارسنجی
        train_size = int(0.8 * len(full_data))
        train_data = full_data[:train_size]
        val_data = full_data[train_size:]

        # ایجاد دیتاست‌ها
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test)

        # ایجاد DataLoader
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
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

if __name__ == "__main__":
    dataset = HardFakeVsRealDataLoader(
        csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=32,
        eval_batch_size=32,
    )
    print(f"Train dataset length: {len(dataset.loader_train.dataset)}")
    print(f"Validation dataset length: {len(dataset.loader_test.dataset)}")
    print(f"Train loader batches: {len(dataset.loader_train)}")
    print(f"Validation loader batches: {len(dataset.loader_test)}")
