import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

# مسیرها و پارامترها
data_dir = '/kaggle/input/hardfakevsrealfaces'
base_model_weights = '/kaggle/input/resnet50-pth/resnet50-19c8e357.pth'  # فرمت PyTorch
teacher_dir = 'teacher_dir'
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

img_height, img_width = 224, 224
batch_size = 32
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# **1. بارگذاری و آماده‌سازی داده‌ها**
df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
df = df.dropna(subset=['label'])  # حذف ردیف‌های با label NaN

def get_image_path(row):
    folder = 'fake' if row['label'] == 'fake' else 'real'
    return os.path.join(data_dir, folder, row['images_id'] + '.jpg')

df['image_path'] = df.apply(get_image_path, axis=1)
df['file_exists'] = df['image_path'].apply(os.path.isfile)
df = df[df['file_exists']]  # حذف ردیف‌هایی که فایل تصویرشان وجود ندارد

# تقسیم داده‌ها به train، validation و test
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.15 / (1 - 0.15), random_state=42, stratify=train_val_df['label'])

# تعریف Dataset سفارشی
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = 0 if self.dataframe.iloc[idx]['label'] == 'fake' else 1
        if self.transform:
            image = self.transform(image)
        return image, label

# تعریف پیش‌پردازش تصاویر
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # معادل width/height_shift_range
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # پیش‌پردازش ResNet
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ایجاد DataLoaderها
train_dataset = CustomDataset(train_df, transform=train_transform)
val_dataset = CustomDataset(val_df, transform=val_test_transform)
test_dataset = CustomDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# **2. ساخت مدل**
# تعریف مدل سفارشی
class CustomResNet(nn.Module):
    def __init__(self, base_model_weights):
        super(CustomResNet, self).__init__()
        # بارگذاری ResNet50 بدون لایه fc
        base_model = models.resnet50(weights=None)
        base_model.load_state_dict(torch.load(base_model_weights, weights_only=True))
        # حذف لایه fc با انتخاب همه لایه‌ها به جز fc
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # غیرفعال کردن آموزش لایه‌های پایه
        for param in self.features.parameters():
            param.requires_grad = False
        # اضافه کردن لایه‌های سفارشی
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # خروجی: (batch_size, 2048, 1, 1)
        self.flatten = nn.Flatten()               # خروجی: (batch_size, 2048)
        self.fc1 = nn.Linear(2048, 1024)          # خروجی: (batch_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1)             # خروجی: (batch_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)  # خروجی: (batch_size, 2048, H, W)
        x = self.pool(x)      # خروجی: (batch_size, 2048, 1, 1)
        x = self.flatten(x)   # خروجی: (batch_size, 2048)
        x = self.fc1(x)       # خروجی: (batch_size, 1024)
        x = self.relu(x)
        x = self.fc2(x)       # خروجی: (batch_size, 1)
        x = self.sigmoid(x)   # خروجی: (batch_size, 1)
        return x

# ایجاد مدل
model = CustomResNet(base_model_weights).to(device)

# **3. تعریف loss و optimizer**
criterion = nn.BCELoss()  # معادل binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# **4. آموزش مدل**
for epoch in range(epochs):
    # آموزش
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # محاسبه دقت آموزشی
        predicted = (outputs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%')

    # اعتبارسنجی
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # محاسبه دقت اعتبارسنجی
            predicted = (outputs > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%')

# **5. ارزیابی**
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.4f}%')

# **6. ذخیره مدل**
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))
