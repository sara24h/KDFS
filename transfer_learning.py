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
    parser = argparse.ArgumentParser(description='Train ResNet50 and MobileNetV2 models for fake vs real face classification.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k'],
                        help='Dataset to use: hardfake, rvf10k, or 140k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file(s)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained models and outputs')
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
        ddp=False
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
        ddp=False
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
        ddp=False
    )
else:
    raise ValueError("Invalid dataset_mode. Choose 'hardfake', 'rvf10k', or '140k'.")

train_loader = dataset.loader_train
val_loader = dataset.loader_val
test_loader = dataset.loader_test

# تعریف مدل‌ها
# ResNet50
resnet50 = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs_resnet = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs_resnet, 1)
resnet50 = resnet50.to(device)

# MobileNetV2
mobilenet_v2 = models.mobilenet_v2(weights='IMAGENET1K_V1')
num_ftrs_mobile = mobilenet_v2.classifier[1].in_features
mobilenet_v2.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(num_ftrs_mobile, 1)
)
mobilenet_v2 = mobilenet_v2.to(device)

# فریز کردن لایه‌ها
# ResNet50
for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.layer4.parameters():
    param.requires_grad = True
for param in resnet50.fc.parameters():
    param.requires_grad = True

# MobileNetV2
for param in mobilenet_v2.parameters():
    param.requires_grad = False
for param in mobilenet_v2.features[18].parameters():  # Last InvertedResidual block
    param.requires_grad = True
for param in mobilenet_v2.classifier.parameters():
    param.requires_grad = True

# تعریف معیار و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()

# بهینه‌ساز برای ResNet50
optimizer_resnet = optim.Adam([
    {'params': resnet50.layer4.parameters(), 'lr': 1e-5},
    {'params': resnet50.fc.parameters(), 'lr': lr}
], weight_decay=1e-4)

# بهینه‌ساز برای MobileNetV2
optimizer_mobile = optim.Adam([
    {'params': mobilenet_v2.features[18].parameters(), 'lr': 1e-5},
    {'params': mobilenet_v2.classifier.parameters(), 'lr': lr}
], weight_decay=1e-4)

# تابع آموزش
def train_model(model, optimizer, model_name):
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
        print(f'{model_name} Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

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
        print(f'{model_name} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return model

# تابع تست
def test_model(model, model_name):
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
    test_loss_avg = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'{model_name} Test Loss: {test_loss_avg:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss_avg, test_accuracy

# تابع نمایش تصاویر نمونه
def visualize_predictions(model, model_name):
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
            print(f"{model_name} Image: {img_path}, True Label: {true_label}, Predicted: {predicted_label}")

    plt.tight_layout()
    file_path = os.path.join(teacher_dir, f'test_samples_{model_name.lower()}.png')
    plt.savefig(file_path)
    display(IPImage(filename=file_path))

# آموزش مدل‌ها
print("Training ResNet50...")
resnet50 = train_model(resnet50, optimizer_resnet, "ResNet50")
print("\nTraining MobileNetV2...")
mobilenet_v2 = train_model(mobilenet_v2, optimizer_mobile, "MobileNetV2")

# تست مدل‌ها
print("\nTesting ResNet50...")
test_model(resnet50, "ResNet50")
print("\nTesting MobileNetV2...")
test_model(mobilenet_v2, "MobileNetV2")

# نمایش تصاویر نمونه
print("\nVisualizing ResNet50 predictions...")
visualize_predictions(resnet50, "ResNet50")
print("\nVisualizing MobileNetV2 predictions...")
visualize_predictions(mobilenet_v2, "MobileNetV2")

# ذخیره مدل‌ها
torch.save(resnet50.state_dict(), os.path.join(teacher_dir, 'teacher_model_resnet50.pth'))
torch.save(mobilenet_v2.state_dict(), os.path.join(teacher_dir, 'teacher_model_mobilenetv2.pth'))

# محاسبه FLOPs و پارامترها
print("\nResNet50 Complexity:")
flops_resnet, params_resnet = get_model_complexity_info(resnet50, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('ResNet50 FLOPs:', flops_resnet)
print('ResNet50 Parameters:', params_resnet)

print("\nMobileNetV2 Complexity:")
flops_mobile, params_mobile = get_model_complexity_info(mobilenet_v2, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('MobileNetV2 FLOPs:', flops_mobile)
print('MobileNetV2 Parameters:', params_mobile)
