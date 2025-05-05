import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import random
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info

# تنظیمات اولیه
dataset_mode = 'hardfake'  # یا 'rvf10k'، بسته به دیتاست شما
data_dir = '/kaggle/input/your-dataset'  # مسیر دیتاست در Kaggle
teacher_dir = '/kaggle/working/teacher_dir'  # مسیر ذخیره‌سازی خروجی‌ها
img_height = 300
img_width = 300
batch_size = 32
epochs = 15
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ایجاد پوشه خروجی در صورت عدم وجود
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# تعریف تبدیل‌ها برای تصاویر
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# فرض می‌کنیم دیتاست شما شامل فایل CSV و تصاویر است
# اگر از Dataset_selector استفاده نمی‌کنید، این بخش را با دیتاست خود جایگزین کنید
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['images_id'])
        image = Image.open(img_name).convert('RGB')
        label = 1 if self.data.iloc[idx]['label'] == 'real' else 0
        if self.transform:
            image = self.transform(image)
        return image, label

# بارگذاری دیتاست
if dataset_mode == 'hardfake':
    train_dataset = FaceDataset(
        csv_file=os.path.join(data_dir, 'data.csv'),
        root_dir=data_dir,
        transform=transform
    )
    test_dataset = train_dataset  # برای ساده‌سازی، از همان دیتاست استفاده می‌کنیم
elif dataset_mode == 'rvf10k':
    train_dataset = FaceDataset(
        csv_file=os.path.join(data_dir, 'train.csv'),
        root_dir=data_dir,
        transform=transform
    )
    test_dataset = FaceDataset(
        csv_file=os.path.join(data_dir, 'valid.csv'),
        root_dir=data_dir,
        transform=transform
    )
else:
    raise ValueError("لطفاً dataset_mode را به 'hardfake' یا 'rvf10k' تنظیم کنید.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# تنظیم مدل ResNet50
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # برای طبقه‌بندی باینری
model = model.to(device)

# فریز کردن لایه‌ها به جز لایه نهایی
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# تعریف تابع هزینه و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

# آموزش مدل
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

# نمایش 10 تصویر تست
model.eval()
test_data = test_dataset.data
random_indices = random.sample(range(len(test_data)), min(10, len(test_data)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = test_data.iloc[idx]
        img_name = row['images_id']
        label_str = row['label']
        img_path = os.path.join(data_dir, img_name)
        if not os.path.exists(img_path):
            print(f"هشدار: تصویر یافت نشد: {img_path}")
            axes[i].set_title("تصویر یافت نشد")
            axes[i].axis('off')
            continue
        image = Image.open(img_path).convert('RGB')
        image_transformed = transform(image).unsqueeze(0).to(device)
        output = model(image_transformed).squeeze(1)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        true_label = 'real' if label_str == 'real' else 'fake'
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"تصویر: {img_path}, لیبل واقعی: {true_label}, پیش‌بینی‌شده: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))

# ذخیره مدل
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))
