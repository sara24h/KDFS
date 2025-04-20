import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class Dataset_hardfakevsreal(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.label_map = {'Fake': 0, 'Real': 1}  # نگاشت برچسب‌ها به اعداد

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # ستون 'id'
        label_str = self.data.iloc[idx, 1]  # ستون 'label'
        label = self.label_map[label_str]

        # تعیین مسیر تصویر (بر اساس برچسب)
        if label_str == 'Fake':
            img_path = os.path.join(self.data_dir, 'fake', img_name)
        else:
            img_path = os.path.join(self.data_dir, 'real', img_name)

        # بارگذاری تصویر
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_loaders(data_dir, csv_file, train_batch_size, eval_batch_size, num_workers, pin_memory, ddp=False):
        # تعریف پیش‌پردازش‌ها برای تصاویر 300×300
        train_transform = transforms.Compose([
            transforms.RandomCrop(300, padding=30),  # برش تصادفی با حفظ اندازه 300×300
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop(300),  # برش مرکزی با حفظ اندازه 300×300
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # بارگذاری دیتاست
        dataset = Dataset_hardfakevsreal(data_dir, csv_file)

        # تقسیم داده‌ها به train و validation (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))  # حدود 1030 تصویر
        val_size = len(dataset) - train_size  # حدود 258 تصویر
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataset.dataset = Dataset_hardfakevsreal(data_dir, csv_file, transform=train_transform)
        val_dataset.dataset = Dataset_hardfakevsreal(data_dir, csv_file, transform=val_transform)

        # ایجاد DataLoader
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, val_loader
