import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from IPython.display import Image as IPImage, display
from data.dataset import FaceDataset
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from ptflops import get_model_complexity_info

# تابع برای解析 کردن آرگومان‌های خط فرمان
def parse_args():
    parser = argparse.ArgumentParser(description='آموزش مدل ResNet برای طبقه‌بندی چهره‌های جعلی و واقعی')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k'],
                        help='حالت مجموعه داده: hardfake یا rvf10k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='مسیر دایرکتوری مجموعه داده که شامل تصاویر و فایل‌های CSV است')
    parser.add_argument('--base_model_weights', type=str, default='/kaggle/input/resnet50-pth/resnet50-19c8e357.pth',
                        help='مسیر وزن‌های از پیش آموزش‌دیده ResNet50')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='دایرکتوری برای ذخیره مدل آموزش‌دیده و خروجی‌ها')
    parser.add_argument('--img_height', type=int, default=300,
                        help='ارتفاع تصاویر ورودی برای مجموعه داده hardfake')
    parser.add_argument('--img_width', type=int, default=300,
                        help='عرض تصاویر ورودی برای مجموعه داده hardfake')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='اندازه دسته برای آموزش')
    parser.add_argument('--epochs', type=int, default=15,
                        help='تعداد دوره‌های آموزش')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='نرخ یادگیری برای بهینه‌ساز')
    return parser.parse_args()

# دریافت آرگومان‌ها
args = parse_args()

# تنظیم متغیرها
dataset_mode = args.dataset_mode
data_dir = args.data_dir
base_model_weights = args.base_model_weights
teacher_dir = args.teacher_dir
image_size = (256, 256) if dataset_mode == 'rvf10k' else (args.img_height, args.img_width)
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# بررسی وجود دایرکتوری‌ها و فایل‌ها
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"دایرکتوری {data_dir} یافت نشد!")
if not os.path.exists(base_model_weights):
    raise FileNotFoundError(f"وزن‌های از پیش آموزش‌دیده {base_model_weights} یافت نشد!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# تابع برای ساخت مسیر کامل تصویر
def create_full_image_path(row, split=None):
    folder = 'fake' if row['label'] == 'fake' else 'real'
    img_name = row['images_id']
    if not img_name.endswith('.jpg'):
        img_name += '.jpg'
    if split:
        return os.path.join(split, folder, img_name)
    else:
        return os.path.join(folder, img_name)

# بارگذاری و تقسیم داده‌ها بر اساس حالت مجموعه داده
if dataset_mode == 'hardfake':
    csv_file = os.path.join(data_dir, 'data.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"فایل CSV {csv_file} یافت نشد!")
    df = pd.read_csv(csv_file)
    df['images_id'] = df.apply(create_full_image_path, axis=1)
    # تقسیم به مجموعه‌های آموزش، اعتبارسنجی و آزمون
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
elif dataset_mode == 'rvf10k':
    train_csv_file = os.path.join(data_dir, 'train.csv')
    valid_csv_file = os.path.join(data_dir, 'valid.csv')
    if not os.path.exists(train_csv_file) or not os.path.exists(valid_csv_file):
        raise FileNotFoundError(f"فایل‌های CSV یافت نشدند!")
    train_df = pd.read_csv(train_csv_file)
    val_df = pd.read_csv(valid_csv_file)
    train_df['images_id'] = train_df.apply(lambda row: create_full_image_path(row, 'train'), axis=1)
    val_df['images_id'] = val_df.apply(lambda row: create_full_image_path(row, 'valid'), axis=1)
    test_df = val_df  # استفاده از valid به عنوان test برای rvf10k
else:
    raise ValueError("حالت مجموعه داده نامعتبر است")

# تعریف تبدیل‌ها برای آموزش و اعتبارسنجی/آزمون
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(image_size[0], padding=8),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ایجاد مجموعه‌های داده
train_dataset = FaceDataset(train_df, data_dir, transform=transform_train)
val_dataset = FaceDataset(val_df, data_dir, transform=transform_test)
test_dataset = FaceDataset(test_df, data_dir, transform=transform_test)

# ایجاد بارگذارهای داده
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# تعریف مدل
model = ResNet_50_hardfakevsreal()
model = model.to(device)

# بارگذاری وزن‌های از پیش آموزش‌دیده
state_dict = torch.load(base_model_weights, weights_only=True)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)

# تنظیم تابع خطا و بهینه‌ساز
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# حلقه آموزش
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f'دوره {epoch+1}, خطای آموزش: {train_loss:.4f}, دقت آموزش: {train_accuracy:.2f}%')
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f'خطای اعتبارسنجی: {val_loss:.4f}, دقت اعتبارسنجی: {val_accuracy:.2f}%')

# ارزیابی روی مجموعه آزمون
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).long()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'خطای آزمون: {test_loss / len(test_loader):.4f}, دقت آزمون: {100 * correct / total:.2f}%')

# نمایش نمونه‌های آزمون
random_indices = random.sample(range(len(test_df)), min(10, len(test_df)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = test_df.iloc[idx]
        img_name = row['images_id']
        label_str = row['label']
        img_path = os.path.join(data_dir, img_name)
        if not os.path.exists(img_path):
            print(f"هشدار: تصویر یافت نشد: {img_path}")
            axes[i].set_title("تصویر یافت نشد")
            axes[i].axis('off')
            continue
        image = Image.open(img_path).convert('RGB')
        image_transformed = transform_test(image).unsqueeze(0).to(device)
        output, _ = model(image_transformed)
        _, predicted = torch.max(output, 1)
        predicted_label = 'واقعی' if predicted.item() == 1 else 'جعلی'
        true_label = 'واقعی' if label_str == 'real' else 'جعلی'
        axes[i].imshow(image)
        axes[i].set_title(f'واقعی: {true_label}\nپیش‌بینی: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"تصویر: {img_path}, برچسب واقعی: {true_label}, پیش‌بینی‌شده: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))

# ذخیره مدل
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

# محاسبه پیچیدگی مدل
flops, params = get_model_complexity_info(model, (3, image_size[0], image_size[1]), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('پارامترها:', params)
