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
        img_name = self.data.iloc[idx]['images_id']
        img_path = os.path.join(self.root_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")
        
        label = self.label_map[self.data.iloc[idx]['label']]
        
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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Load and validate CSV
        full_data = pd.read_csv(csv_file)
        print(f"Initial dataset size: {len(full_data)}")
        
        # Construct full image paths
        def create_full_image_path(row):
            folder = 'fake' if row['label'] == 'fake' else 'real'
            img_name = row['images_id']
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            return os.path.join(folder, img_name)
        
        full_data['images_id'] = full_data.apply(create_full_image_path, axis=1)
        
        # Filter out missing files
        def file_exists(row):
            img_path = os.path.join(root_dir, row['images_id'])
            exists = os.path.exists(img_path)
            if not exists:
                print(f"Missing file: {img_path}")
            return exists
        
        full_data = full_data[full_data.apply(file_exists, axis=1)]
        print(f"Filtered dataset size: {len(full_data)}")
        
        # Save filtered CSV for debugging
        filtered_csv = '/kaggle/working/filtered_data.csv'
        full_data.to_csv(filtered_csv, index=False)
        print(f"Filtered CSV saved to: {filtered_csv}")
        
        # Shuffle and split data
        full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(0.8 * len(full_data))
        train_data = full_data[:train_size]
        val_data = full_data[train_size:]

        # Create FaceDataset instances
        self.train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        self.val_dataset = FaceDataset(val_data, root_dir, transform=transform_test)

        # Create data loaders
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            )
            self.loader_train = DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                self.train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=pin_memory,
            )

        self.loader_test = DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging
            pin_memory=pin_memory,
        )
        
        print(f"Train loader length: {len(self.loader_train)}")
        print(f"Validation loader length: {len(self.loader_test)}")

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        return self.train_dataset[idx]
