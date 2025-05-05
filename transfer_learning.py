import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, dataset_mode, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_mode = dataset_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.dataset_mode == 'hardfake':
            img_name = os.path.join(self.root_dir, self.data.iloc[idx]['images_id'])
            label = self.data.iloc[idx]['label']
        else:  # rvf10k or 140k
            img_name = os.path.join(self.root_dir, self.data.iloc[idx]['path'])
            label = 1 if self.data.iloc[idx]['label'] == 'real' else 0

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DatasetSelector:
    def __init__(self, dataset_mode, data_dir, train_batch_size, eval_batch_size, num_workers=4, pin_memory=True):
        self.dataset_mode = dataset_mode
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # تنظیم اندازه تصویر
        self.img_size = 300 if dataset_mode == 'hardfake' else 256

        # تعریف تبدیل‌ها
        self.transform_train = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # تنظیم دیتاست‌ها
        if dataset_mode == 'hardfake':
            self.loader_train = self._create_loader(
                csv_file=os.path.join(data_dir, 'data.csv'),
                root_dir=data_dir,
                transform=self.transform_train,
                is_train=True
            )
            self.loader_val = self._create_loader(
                csv_file=os.path.join(data_dir, 'data.csv'),
                root_dir=data_dir,
                transform=self.transform_eval,
                is_train=False
            )
            self.loader_test = self.loader_val
        elif dataset_mode == 'rvf10k':
            self.loader_train = self._create_loader(
                csv_file=os.path.join(data_dir, 'train.csv'),
                root_dir=data_dir,
                transform=self.transform_train,
                is_train=True
            )
            self.loader_val = self._create_loader(
                csv_file=os.path.join(data_dir, 'valid.csv'),
                root_dir=data_dir,
                transform=self.transform_eval,
                is_train=False
            )
            self.loader_test = self.loader_val
        elif dataset_mode == '140k':
            self.loader_train = self._create_loader(
                csv_file=os.path.join(data_dir, 'train.csv'),
                root_dir=os.path.join(data_dir, 'real_vs_fake/real-vs-fake'),
                transform=self.transform_train,
                is_train=True
            )
            self.loader_val = self._create_loader(
                csv_file=os.path.join(data_dir, 'valid.csv'),
                root_dir=os.path.join(data_dir, 'real_vs_fake/real-vs-fake'),
                transform=self.transform_eval,
                is_train=False
            )
            self.loader_test = self._create_loader(
                csv_file=os.path.join(data_dir, 'test.csv'),
                root_dir=os.path.join(data_dir, 'real_vs_fake/real-vs-fake'),
                transform=self.transform_eval,
                is_train=False
            )

    def _create_loader(self, csv_file, root_dir, transform, is_train):
        dataset = FaceDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            dataset_mode=self.dataset_mode,
            transform=transform
        )
        batch_size = self.train_batch_size if is_train else self.eval_batch_size
        shuffle = is_train
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

def preceding section

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model for fake vs real face classification.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k'],
                        help='Dataset to use: hardfake, rvf10k, or 140k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file(s)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_args()

# تنظیم پارامترها
dataset_mode = args.dataset_mode
data_dir = args.data_dir
teacher_dir = args.teacher_dir
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# بررسی وجود دایرکتوری‌ها
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# بارگذاری داده‌ها
dataset = DatasetSelector(
    dataset_mode=dataset_mode,
    data_dir=data_dir,
    train_batch_size=batch_size,
    eval_batch_size=batch_size
)

train_loader = dataset.loader_train
val_loader = dataset.loader_val
test_loader = dataset.loader_test

# تعریف مدل
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# فریز کردن لایه‌ها
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# تعریف معیار و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': lr}
], weight_decay=1e-4)

# حلقه آموزش
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # اعتبارسنجی
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# تست
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

# نمایش تصاویر نمونه
val_data = dataset.loader_test.dataset.data
transform_test = dataset.loader_test.dataset.transform

random_indices = random.sample(range(len(val_data)), min(10, len(val_data)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = val_data.iloc[idx]
        img_column = 'path' if dataset_mode in ['rvf10k', '140k'] else 'images_id'
        img_name = row[img_column]
        label = row['label']
        img_path = os.path.join(dataset.loader_test.dataset.root_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            axes[i].set_title("Image not found")
            axes[i].axis('off')
            continue
        image = Image.open(img_path).convert('RGB')
        image_transformed = transform_test(image).unsqueeze(0).to(device)
        output = model(image_transformed).squeeze(1)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        true_label = 'real' if label == 1 else 'fake'
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"Image: {img_path}, True Label: {true_label}, Predicted: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))

# ذخیره مدل
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

# محاسبه FLOPs و پارامترها
img_size = 300 if dataset_mode == 'hardfake' else 256
flops, params = get_model_complexity_info(model, (3, img_size, img_size), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
