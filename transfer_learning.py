import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import PIL.Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet-based model for fake vs real face classification.')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/hardfakevsrealfaces')
    parser.add_argument('--base_model_weights', type=str, default='/kaggle/input/resnet50-pth/resnet50-19c8e357.pth')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir')
    parser.add_argument('--img_height', type=int, default=224) 
    parser.add_argument('--img_width', type=int, default=224)  
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0001)
    return parser.parse_args()


args = parse_args()

data_dir = args.data_dir
base_model_weights = args.base_model_weights
teacher_dir = args.teacher_dir
img_height = args.img_height
img_width = args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)


csv_file = os.path.join(data_dir, 'data.csv')
df = pd.read_csv(csv_file)
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])


test_csv_file = os.path.join(teacher_dir, 'test_data.csv')
test_df.to_csv(test_csv_file, index=False)

dataset = Dataset_hardfakevsreal(
    csv_file=csv_file,
    root_dir=data_dir,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    ddp=False
)


train_loader = dataset.loader_train
val_loader = dataset.loader_test


test_dataset = FaceDataset(
    data_frame=test_df,
    root_dir=data_dir,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


model = ResNet_50_hardfakevsreal()
model = model.to(device)


state_dict = torch.load(base_model_weights, weights_only=True)
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
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%')


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
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%')


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
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.4f}%')


random_indices = random.sample(range(len(test_df)), 10)
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = test_df.iloc[idx]
        img_name = row['images_id']
        label_str = row['label']
        
        if not img_name.endswith('.jpg'):
            img_name = img_name + '.jpg'
        
        folder = 'fake' if label_str == 'fake' else 'real'
        img_path = os.path.join(data_dir, folder, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            axes[i].set_title("Image not found")
            axes[i].axis('off')
            continue
        
        image = PIL.Image.open(img_path).convert('RGB')
        image_transformed = test_dataset.transform(image).unsqueeze(0).to(device)
        
        output, _ = model(image_transformed)
        _, predicted = torch.max(output, 1)
        predicted_label = 'real' if predicted.item() == 1 else 'fake'
        true_label = 'real' if label_str == 'real' else 'fake'
        
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        
        print(f"Image: {img_path}, True Label: {true_label}, Predicted: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
print(f"saved in {file_path}")

display(IPImage(filename=file_path))


torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model.pth'))

flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
