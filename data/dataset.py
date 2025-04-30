import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    """Dataset class for loading face images with real/fake labels."""
    
    def __init__(self, data_frame, root_dir, transform=None):
        """
        Initialize the FaceDataset.
        
        Args:
            data_frame (pd.DataFrame): DataFrame containing image IDs and labels.
            root_dir (str): Root directory containing the image files.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'real': 1, 'fake': 0}

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the class index, or (None, None) if image is missing.
        """
        img_name = os.path.join(self.root_dir, self.data['images_id'].iloc[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {img_name}, skipping...")
            return None, None
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None
        label = self.data['label'].iloc[idx]
        if isinstance(label, str):
            label = self.label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

def filter_valid_images(df, root_dir):
    """Filter out rows where images are missing."""
    valid_rows = []
    for idx, row in df.iterrows():
        img_path = os.path.join(root_dir, row['images_id'])
        if os.path.exists(img_path):
            valid_rows.append(row)
        else:
            print(f"Skipping missing image: {img_path}")
    filtered_df = pd.DataFrame(valid_rows)
    print(f"Filtered dataset size: {len(filtered_df)} (original: {len(df)})")
    return filtered_df

def collate_fn(batch):
    """Custom collate function to skip None items in a batch."""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class FakeVsReal10kDataset:
    """Dataset class for the RVF10k dataset, using separate train and valid CSV files and creating a test split."""
    
    def __init__(
        self,
        train_csv_file,
        valid_csv_file,
        root_dir,
        batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
        test_split_ratio=0.33,
    ):
        """
        Initialize the RVF10k dataset.
        
        Args:
            train_csv_file (str): Path to the training CSV file.
            valid_csv_file (str): Path to the validation CSV file.
            root_dir (str): Root directory containing the image files.
            batch_size (int): Batch size for training, validation, and test.
            num_workers (int): Number of workers for data loading.
            pin_memory (bool): Whether to use pinned memory for faster GPU transfer.
            ddp (bool): Whether to use Distributed Data Parallel (DDP).
            test_split_ratio (float): Ratio of validation data to use as test data.
        """
        # Define transformations
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

        # Load CSV files
        train_data = pd.read_csv(train_csv_file).rename(columns={'id': 'images_id'})
        valid_data = pd.read_csv(valid_csv_file).rename(columns={'id': 'images_id'})

        # Split validation data into validation and test
        valid_data, test_data = train_test_split(
            valid_data,
            test_size=test_split_ratio,
            stratify=valid_data['label'],
            random_state=3407
        )
        train_data = train_data.reset_index(drop=True)
        valid_data = valid_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # Modify image paths
        def create_full_image_path(row, folder_type):
            folder = 'real' if row['label'] == 1 else 'fake'
            img_name = os.path.basename(row['images_id'])
            possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
            for ext in possible_extensions:
                candidate_name = img_name if img_name.endswith(ext) else img_name + ext
                full_path = os.path.join(folder_type, folder, candidate_name)
                absolute_path = os.path.join(root_dir, full_path)
                if os.path.exists(absolute_path):
                    return full_path
            print(f"Warning: No valid image found for {img_name}")
            return full_path

        train_data['images_id'] = train_data.apply(
            lambda row: create_full_image_path(row, 'train'), axis=1
        )
        valid_data['images_id'] = valid_data.apply(
            lambda row: create_full_image_path(row, 'valid'), axis=1
        )
        test_data['images_id'] = test_data.apply(
            lambda row: create_full_image_path(row, 'valid'), axis=1
        )

        # Filter out missing images
        train_data = filter_valid_images(train_data, root_dir)
        valid_data = filter_valid_images(valid_data, root_dir)
        test_data = filter_valid_images(test_data, root_dir)

        # Debug: Print dataset statistics
        for name, data in [('Train', train_data), ('Validation', valid_data), ('Test', test_data)]:
            print(f"\n{name} dataset statistics:")
            print("Sample image paths:", data['images_id'].head().tolist())
            print(f"Total size: {len(data)}")
            print(f"Duplicate rows: {data.duplicated().sum()}")
            print(f"Label distribution:\n{data['label'].value_counts()}")
            print(f"Unique labels: {data['label'].unique()}")

        # Create datasets
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train)
        valid_dataset = FaceDataset(valid_data, root_dir, transform=transform_eval)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_eval)

        # Create data loaders
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
            self.loader_train = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
                collate_fn=collate_fn,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )

        self.loader_valid = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        self.loader_test = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        # Debug: Print loader information
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
        batch_size=32,
        test_split_ratio=0.33,
    )
