import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
try:
    from IPython.display import Image as IPImage, display
except ImportError:
    IPImage = None
from ptflops import get_model_complexity_info

# کلاس دیتاست سفارشی
class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['images_id'])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# انتخاب‌گر دیتاست
class Dataset_selector:
    def __init__(self, dataset_mode, hardfake_csv_file=None, hardfake_root_dir=None,
                 rvf10k_train_csv=None, rvf10k_valid_csv=None, rvf10k_root_dir=None,
                 realfake140k_train_csv=None, realfake140k_valid_csv=None, realfake140k_test_csv=None,
                 realfake140k_root_dir=None, train_batch_size=32, eval_batch_size=32,
                 num_workers=4, pin_memory=True, ddp=False):
        self.dataset_mode = dataset_mode
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # تعریف تبدیل‌ها
        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_mode == 'hardfake':
            self.loader_train = DataLoader(
                FaceDataset(hardfake_csv_file, hardfake_root_dir, self.transform_train),
                batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_val = DataLoader(
                FaceDataset(hardfake_csv_file, hardfake_root_dir, self.transform_eval),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_test = self.loader_val  # برای hardfake، تست و اعتبارسنجی یکسان هستند
        else:
            raise ValueError("Only 'hardfake' dataset_mode is implemented in this code.")

# تابع برای پار싱 آرگومان‌ها
def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model for fake vs real face classification.')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'],
                        help='Model to use (default: resnet50)')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake'],
                        help='Dataset to use: hardfake')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file')
    parser.add_argument('--teacher_dir', type=str, default='/kaggle/working/teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=256,
                        help='Height of input images')
    parser.add_argument('--img_width', type=int, default=256,
                        help='Width of input images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    return parser.parse_args()

def main():
    args = parse_args()

    # تنظیم پارامترها
    dataset_mode = args.dataset_mode
    data_dir = args.data_dir
    teacher_dir = args.teacher_dir
    img_height = args.img_height
    img_width = args.img_width
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
            pin_memory=True
        )
    else:
        raise ValueError("Invalid dataset_mode. Choose 'hardfake'.")

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

    # نمایش 20 تصویر نمونه
    val_data = dataset.loader_test.dataset.data
    transform_test = dataset.loader_test.dataset.transform

    # انتخاب تصادفی 20 شاخص معتبر
    valid_indices = []
    while len(valid_indices) < 20 and len(valid_indices) < len(val_data):
        idx = random.randint(0, len(val_data)-1)
        row = val_data.iloc[idx]
        img_name = row['images_id']
        img_path = os.path.join(data_dir, img_name)
        if os.path.exists(img_path) and idx not in valid_indices:
            valid_indices.append(idx)

    if len(valid_indices) < 20:
        print(f"Warning: Only {len(valid_indices)} valid images found.")

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()

    with torch.no_grad():
        for i, idx in enumerate(valid_indices):
            row = val_data.iloc[idx]
            img_name = row['images_id']
            label = row['label']
            img_path = os.path.join(data_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axes[i].set_title("Image not found")
                axes[i].axis('off')
                continue
            image_transformed = transform_test(image).unsqueeze(0).to(device)
            output = model(image_transformed).squeeze(1)
            prob = torch.sigmoid(output).item()
            predicted_label = 'real' if prob > 0.5 else 'fake'
            true_label = 'real' if label == 1 else 'fake'
            axes[i].imshow(image)
            axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}\nProb: {prob:.2f}', fontsize=8)
            axes[i].axis('off')
            print(f"Image {i+1}: {img_path}, True Label: {true_label}, Predicted: {predicted_label}, Probability: {prob:.2f}")

    # خاموش کردن محورهای اضافی
    for i in range(len(valid_indices), 20):
        axes[i].axis('off')

    plt.tight_layout()
    file_path = os.path.join(teacher_dir, 'test_samples_20.png')
    plt.savefig(file_path)
    print(f"Saved 20 sample images to {file_path}")

    # نمایش تصاویر در محیط‌های مختلف
    try:
        if IPImage:
            display(IPImage(filename=file_path))  # برای Jupyter/Kaggle
        else:
            plt.show()  # برای محیط‌های غیر-Jupyter
    except Exception as e:
        print(f"Error displaying image: {e}. Please check the saved file at {file_path}")

    # ذخیره مدل
    torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

    # محاسبه FLOPs و پارامترها
    flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
    print('FLOPs:', flops)
    print('Parameters:', params)

if __name__ == '__main__':
    main()
