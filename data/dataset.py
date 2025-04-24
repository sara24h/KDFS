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
        self.label_map = {'fake': 0, 'real': 1}  # مطابق با فایل CSV

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['images_id']
        label_str = self.data.iloc[idx]['label']
        
        if label_str not in self.label_map:
            raise KeyError(f"Label '{label_str}' not found in label_map. Available labels: {list(self.label_map.keys())}")
        
        label = self.label_map[label_str]

        # افزودن پسوند .jpg اگر لازم باشد
        if not img_name.endswith('.jpg'):
            img_name = img_name + '.jpg'

        if label_str == 'fake':
            img_path = os.path.join(self.data_dir, 'fake', img_name)
        else:
            img_path = os.path.join(self.data_dir, 'real', img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_val_test_transform():
        """Returns the validation/test transform as a static method."""
        return transforms.Compose([
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_loaders(data_dir, csv_file, train_batch_size, eval_batch_size, num_workers, pin_memory, ddp=False, test_csv_file=None):
        """Returns train, validation, and test DataLoaders."""
        train_transform = transforms.Compose([
            transforms.RandomCrop(300, padding=30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_test_transform = Dataset_hardfakevsreal.get_val_test_transform()

        # Create dataset
        dataset = Dataset_hardfakevsreal(data_dir, csv_file)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Apply transforms to datasets
        train_dataset.dataset = Dataset_hardfakevsreal(data_dir, csv_file, transform=train_transform)
        val_dataset.dataset = Dataset_hardfakevsreal(data_dir, csv_file, transform=val_test_transform)

        # Create DataLoaders
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

        test_loader = None
        if test_csv_file:
            test_dataset = Dataset_hardfakevsreal(data_dir, test_csv_file, transform=val_test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

        return train_loader, val_loader, test_loader
