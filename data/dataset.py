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

       
        full_data = pd.read_csv(csv_file)
        full_data = full_data.sample(frac=1).reset_index(drop=True)

        train_size = int(0.8 * len(full_data))
        train_data = full_data[:train_size]
        val_data = full_data[train_size:]

        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test)

     
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


if __name__ == "__main__":
    dataset = Dataset_hardfakevsreal(
        csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=32,
        eval_batch_size=32,
    )
    print(f"Train loader length: {len(dataset.loader_train)}")
    print(f"Validation loader length: {len(dataset.loader_test)}")
