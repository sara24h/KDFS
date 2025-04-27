import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch

class Dataset_hardfakevsreal(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.label_map = {'fake': 0, 'real': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['images_id']
        label_str = self.data.iloc[idx]['label']
        
        if label_str not in self.label_map:
            raise KeyError(f"Label '{label_str}' not found in label_map.")
        label = self.label_map[label_str]

        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        subdir = 'fake' if label_str == 'fake' else 'real'
        img_path = os.path.join(self.data_dir, subdir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def get_train_transform():
        return transforms.Compose([
            transforms.RandomCrop(300, padding=30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_test_transform():
        return transforms.Compose([
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_loaders(data_dir, csv_file,
                    train_batch_size, eval_batch_size,
                    num_workers, pin_memory,
                    ddp=False, test_csv_file=None, seed=42):
        # تعریف ترنسفورم‌ها
        train_tf = Dataset_hardfakevsreal.get_train_transform()
        val_tf = Dataset_hardfakevsreal.get_val_test_transform()

        # دیتاست پایه (بدون ترنسفورم)
        base_dataset = Dataset_hardfakevsreal(data_dir, csv_file, transform=None)
        num_samples = len(base_dataset)

        # شافل تصادفی ایندکس‌ها
        torch.manual_seed(seed)
        indices = torch.randperm(num_samples).tolist()
        split = int(0.8 * num_samples)
        train_idx, val_idx = indices[:split], indices[split:]

        # ساخت دیتاست با ترنسفورم و زیرمجموعه‌ی ایندکس‌ها
        train_dataset = Subset(Dataset_hardfakevsreal(data_dir, csv_file, transform=train_tf), train_idx)
        val_dataset   = Subset(Dataset_hardfakevsreal(data_dir, csv_file, transform=val_tf),   val_idx)

        # Optional چاپ اندازه‌ها برای دیباگ
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # DataLoaderها
        if ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                      sampler=sampler, num_workers=num_workers,
                                      pin_memory=pin_memory)
        else:
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_memory)

        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)

        test_loader = None
        if test_csv_file:
            test_ds = Dataset_hardfakevsreal(data_dir, test_csv_file, transform=val_tf)
            test_loader = DataLoader(test_ds, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
