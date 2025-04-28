import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold

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
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class Dataset_hardfakevsreal:
    def __init__(
        self,
        csv_file,
        root_dir,
        train_batch_size,
        eval_batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
        n_splits=5,
    ):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ddp = ddp
        self.n_splits = n_splits

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])

        self.transform_test = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.full_data = pd.read_csv(csv_file)
        self.full_data = self.full_data.sample(frac=1, random_state=3407).reset_index(drop=True)
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=3407)

    def get_fold_dataloaders(self, fold_idx):
        train_idx, val_idx = list(self.kf.split(self.full_data))[fold_idx]
        train_data = self.full_data.iloc[train_idx]
        val_data = self.full_data.iloc[val_idx]

        train_dataset = FaceDataset(train_data, self.root_dir, transform=self.transform_train)
        val_dataset = FaceDataset(val_data, self.root_dir, transform=self.transform_test)

        if self.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=train_sampler,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_loader, val_loader
