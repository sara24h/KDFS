import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from dataset import FaceDataset, FakeVsReal10kDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet-based model for fake vs real face classification.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing images and CSV file')
    parser.add_argument('--dataset_type', type=str, choices=['hardfakevsreal', 'rvf10k'], default='rvf10k', help='Type of dataset to use: hardfakevsreal or rvf10k')
    parser.add_argument('--csv_file', type=str, default='data.csv', help='Name of the CSV file in data_dir (default: data.csv, used for hardfakevsreal)')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='Name of the train CSV file in data_dir (default: train.csv, used for rvf10k)')
    parser.add_argument('--valid_csv', type=str, default='valid.csv', help='Name of the valid CSV file in data_dir (default: valid.csv, used for rvf10k)')
    parser.add_argument('--base_model_weights', type=str, default='/kaggle/input/resnet50-pth/resnet50-19c8e357.pth', help='Path to the pretrained ResNet50 weights')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir', help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=300, help='Height of input images')
    parser.add_argument('--img_width', type=int, default=300, help='Width of input images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training, validation, and test')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_args()

data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = args.img_height
img_width = args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU")

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

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

    def filter_valid_images(df, root_dir):
        valid_rows = []
        for idx, row in df.iterrows():
            img_path = os.path.join(root_dir, row['images_id'])
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                print(f"Skipping missing image: {img_path}")
        return pd.DataFrame(valid_rows)

    train_df = filter_valid_images(train_df, data_dir)
    val_df = filter_valid_images(val_df, data_dir)
    test_df = filter_valid_images(test_df, data_dir)

    train_dataset = FaceDataset(train_df, data_dir, transform=transform_train)
    val_dataset = FaceDataset(val_df, data_dir, transform=transform_test)
    test_dataset = FaceDataset(test_df, data_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

elif args.dataset_type == 'rvf10k':
    dataset = FakeVsReal10kDataset(
        train_csv_file=os.path.join(data_dir, args.train_csv),
        valid_csv_file=os.path.join(data_dir, args.valid_csv),
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        test_split_ratio=0.33,
    )
    train_loader = dataset.loader_train
    val_loader = dataset.loader_valid
    test_loader = dataset.loader_test

class ResNet_50_hardfakevsreal(nn.Module):
    def __init__(self):
        super(ResNet_50_hardfakevsreal, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    
    def forward(self, x):
        features = self.resnet(x)
        return features, features

model = ResNet_50_hardfakevsreal()
model = model.to(device)

print(f"Model is on: {next(model.parameters()).device}")

if not os.path.exists(args.base_model_weights):
    raise FileNotFoundError(f"Pretrained weights {args.base_model_weights} not found!")

state_dict = torch.load(args.base_model_weights, map_location='cpu', weights_only=True)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_loader):
        if data is None:
            continue
        images, labels = data
        images, labels = images.to(device), labels.to(device).long()
        if i == 0:
            print(f"Epoch {epoch+1}, Batch 1 - Images are on: {images.device}")
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
        for data in val_loader:
            if data is None:
                continue
            images, labels = data
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

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        if data is None:
            continue
        images, labels = data
        images, labels = images.to(device), labels.to(device).long()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

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

torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

from ptflops import get_model_complexity_info
flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
