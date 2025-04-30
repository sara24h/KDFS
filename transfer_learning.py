import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet-based model for fake vs real face classification.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file')
    parser.add_argument('--dataset_type', type=str, choices=['hardfakevsreal', 'rvf10k'], default='rvf10k',
                        help='Type of dataset to use: hardfakevsreal or rvf10k')
    parser.add_argument('--csv_file', type=str, default='data.csv',
                        help='Name of the CSV file in data_dir (default: data.csv, used for hardfakevsreal)')
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help='Name of the train CSV file in data_dir (default: train.csv, used for rvf10k)')
    parser.add_argument('--valid_csv', type=str, default='valid.csv',
                        help='Name of the valid CSV file in data_dir (default: valid.csv, used for rvf10k)')
    parser.add_argument('--base_model_weights', type=str, default='/kaggle/input/resnet50-pth/resnet50-19c8e357.pth',
                        help='Path to the pretrained ResNet50 weights')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=300,
                        help='Height of input images')
    parser.add_argument('--img_width', type=int, default=300,
                        help='Width of input images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training, validation, and test')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_args()

data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = args.img_height
img_width = args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# Define transforms
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

transform_test = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Dataset class
class FaceDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'real': 1, 'fake': 0}  # For string labels

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data['images_id'].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.data['label'].iloc[idx]
        if isinstance(label, str):
            label = self.label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

if args.dataset_type == 'hardfakevsreal':
    csv_file = os.path.join(data_dir, args.csv_file)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} not found!")
    
    df = pd.read_csv(csv_file)
    image_id_column = 'images_id'
    if image_id_column not in df.columns:
        raise ValueError(f"Column '{image_id_column}' not found in CSV file. Available columns: {df.columns}")
    
    def create_full_image_path(row):
        folder = 'fake' if row['label'] == 'fake' else 'real'
        img_name = row[image_id_column]
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        full_path = os.path.join(data_dir, folder, img_name)
        if not os.path.exists(full_path):
            print(f"Warning: Image not found: {full_path}")
        return os.path.join(folder, img_name)
    df[image_id_column] = df.apply(create_full_image_path, axis=1)

    train_csv_file = os.path.join(teacher_dir, 'train_data.csv')
    df.to_csv(train_csv_file, index=False)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    val_csv_file = os.path.join(teacher_dir, 'val_data.csv')
    test_csv_file = os.path.join(teacher_dir, 'test_data.csv')
    val_df.to_csv(val_csv_file, index=False)
    test_df.to_csv(test_csv_file, index=False)

    train_dataset = FaceDataset(train_df, data_dir, transform=transform_train)
    val_dataset = FaceDataset(val_df, data_dir, transform=transform_test)
    test_dataset = FaceDataset(test_df, data_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

elif args.dataset_type == 'rvf10k':
    train_csv_file = os.path.join(data_dir, args.train_csv)
    valid_csv_file = os.path.join(data_dir, args.valid_csv)
    if not os.path.exists(train_csv_file):
        raise FileNotFoundError(f"Train CSV file {train_csv_file} not found!")
    if not os.path.exists(valid_csv_file):
        raise FileNotFoundError(f"Valid CSV file {valid_csv_file} not found!")
    
    train_df = pd.read_csv(train_csv_file).rename(columns={'id': 'images_id'})
    valid_df = pd.read_csv(valid_csv_file).rename(columns={'id': 'images_id'})
    
    def create_full_image_path(row, folder_type):
        folder = 'real' if row['label'] == 1 else 'fake'
        img_name = os.path.basename(row['images_id'])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        return os.path.join(folder_type, folder, img_name)

    train_df['images_id'] = train_df.apply(lambda row: create_full_image_path(row, 'train'), axis=1)
    valid_df['images_id'] = valid_df.apply(lambda row: create_full_image_path(row, 'valid'), axis=1)

    val_df, test_df = train_test_split(valid_df, test_size=0.33, random_state=42, stratify=valid_df['label'])

    train_dataset = FaceDataset(train_df, data_dir, transform=transform_train)
    val_dataset = FaceDataset(val_df, data_dir, transform=transform_test)
    test_dataset = FaceDataset(test_df, data_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Simplified ResNet model (replace with actual ResNet_50_hardfakevsreal if available)
class ResNet_50_hardfakevsreal(nn.Module):
    def __init__(self):
        super(ResNet_50_hardfakevsreal, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    
    def forward(self, x):
        features = self.resnet(x)
        return features, features

model = ResNet_50_hardfakevsreal()
model = model.to(device)
state_dict = torch.load(args.base_model_weights, map_location=device, weights_only=True)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
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
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

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
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Test evaluation
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
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

# Display test samples
random_indices = random.sample(range(len(test_loader.dataset)), min(10, len(test_loader.dataset)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        image, label = test_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        label_str = 'real' if label == 1 else 'fake'
        
        output, _ = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = 'real' if predicted.item() == 1 else 'fake'
        
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        axes[i].imshow(image_np)
        axes[i].set_title(f'True: {label_str}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"Sample {i+1}, True Label: {label_str}, Predicted: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))

# Save model
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

# Model complexity
from ptflops import get_model_complexity_info
flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
