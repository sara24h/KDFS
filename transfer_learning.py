import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info

from data.dataset import FaceDataset, Dataset_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model with single output for fake vs real face classification with 5-fold cross-validation.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k'],
                        help='Dataset to use: hardfake, rvf10k, or 140k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file(s)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=300,
                        help='Height of input images (default: 300 for hardfake, 256 for rvf10k and 140k)')
    parser.add_argument('--img_width', type=int, default=300,
                        help='Width of input images (default: 300 for hardfake, 256 for rvf10k and 140k)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    return parser.parse_args()

args = parse_args()

# تنظیم پارامترها
dataset_mode = args.dataset_mode
data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_height
img_width = 256 if dataset_mode in ['rvf10k', '140k'] else args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
n_folds = args.n_folds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# بررسی وجود دایرکتوری‌ها
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# بارگذاری داده‌ها
if dataset_mode == 'hardfake':
    dataset = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file=os.path.join(data_dir, 'data.csv'),
        hardfake_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        n_folds=n_folds
    )
elif dataset_mode == 'rvf10k':
    dataset = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv=os.path.join(data_dir, 'train.csv'),
        rvf10k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        rvf10k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        n_folds=n_folds
    )
elif dataset_mode == '140k':
    dataset = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv=os.path.join(data_dir, 'train.csv'),
        realfake140k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        realfake140k_test_csv=os.path.join(data_dir, 'test.csv'),
        realfake140k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        n_folds=n_folds
    )
else:
    raise ValueError("Invalid dataset_mode. Choose 'hardfake', 'rvf10k', or '140k'.")

# تعریف مدل
def initialize_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
    # فریز کردن لایه‌ها
    for param in model.parameters():
        param.requires_grad = False
    # آزاد کردن لایه‌های layer4 و fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

# تعریف معیار و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()

# لیست برای ذخیره معیارهای هر فولد
fold_train_losses = []
fold_train_accuracies = []
fold_val_losses = []
fold_val_accuracies = []

# حلقه روی فولدها
for fold, (train_loader, val_loader) in enumerate(dataset.fold_loaders):
    print(f'\nTraining Fold {fold + 1}/{n_folds}')
    
    # مقداردهی اولیه مدل برای هر فولد
    model = initialize_model()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    # حلقه آموزش برای هر فولد
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
        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

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
        print(f'Fold {fold + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # ذخیره معیارهای این فولد
    fold_train_losses.append(train_loss)
    fold_train_accuracies.append(train_accuracy)
    fold_val_losses.append(val_loss)
    fold_val_accuracies.append(val_accuracy)

    # ذخیره مدل برای این فولد
    torch.save(model.state_dict(), os.path.join(teacher_dir, f'teacher_model_fold_{fold + 1}.pth'))

# گزارش میانگین معیارها
print('\nCross-Validation Results:')
print(f'Average Train Loss: {np.mean(fold_train_losses):.4f} ± {np.std(fold_train_losses):.4f}')
print(f'Average Train Accuracy: {np.mean(fold_train_accuracies):.2f}% ± {np.std(fold_train_accuracies):.2f}%')
print(f'Average Validation Loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}')
print(f'Average Validation Accuracy: {np.mean(fold_val_accuracies):.2f}% ± {np.std(fold_val_accuracies):.2f}%')

# تست با بهترین مدل (مثلاً آخرین فولد)
model.load_state_dict(torch.load(os.path.join(teacher_dir, f'teacher_model_fold_{n_folds}.pth')))
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataset.loader_test:
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f'Test Loss: {test_loss / len(dataset.loader_test):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

# نمایش تصاویر نمونه
val_data = dataset.loader_test.dataset.data
transform_test = dataset.loader_test.dataset.transform

random_indices = random.sample(range(len(val_data)), min(10, len(val_data)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = val_data.iloc[idx]
        img_column = 'path' if dataset_mode == '140k' else 'images_id'
        img_name = row[img_column]
        label = row['label']
        img_path = os.path.join(data_dir, 'real_vs_fake', 'real-vs-fake', img_name) if dataset_mode == '140k' else os.path.join(data_dir, img_name)
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

# محاسبه FLOPs و پارامترها
flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
