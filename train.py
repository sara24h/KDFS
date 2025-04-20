import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from torchvision.models import ResNet50_Weights

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # Base directory containing real and fake subdirs
        self.transform = transform
        self.label_map = {"fake": 0, "real": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx]["label"]  # 'fake' or 'real'
        img_name = os.path.join(self.root_dir, label, f"{self.data.iloc[idx]['images_id']}.jpg")
        image = Image.open(img_name).convert("RGB")
        label_numeric = self.label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label_numeric

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = "/kaggle/input/hardfakevsrealfaces"
csv_file = os.path.join(dataset_dir, "data.csv")
root_dir = dataset_dir  # Points to directory with real and fake subdirs

# Define transforms
transform_train = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Load teacher model
teacher = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Updated to use weights
num_ftrs = teacher.fc.in_features
teacher.fc = nn.Linear(num_ftrs, 2)  # 2 classes: fake and real
teacher = teacher.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Fine-tuning
num_epochs = 10
teacher.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = teacher(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
teacher.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = teacher(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save fine-tuned model
torch.save({"state_dict": teacher.state_dict()}, "teacher_resnet50_finetuned.pth")
print("Teacher model saved to 'teacher_resnet50_finetuned.pth'")


##############


import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.models import ResNet50_Weights

# Base FLOPs for models
Flops_baselines = {
    "resnet_56": 125.49,  # For CIFAR-10 (32x32)
    "ResNet_50": 4134,    # Approximate for ImageNet (224x224)
}

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # Base directory containing real and fake subdirs
        self.transform = transform
        self.label_map = {"fake": 0, "real": 1}  # Fake=0, Real=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx]["label"]
        img_name = os.path.join(self.root_dir, label, f"{self.data.iloc[idx]['images_id']}.jpg")
        image = Image.open(img_name).convert("RGB")
        numeric_label = self.label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, numeric_label

# Training class
class Train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        self.compress_rate = args.compress_rate
        self.resume = args.resume

        self.start_epoch = 0
        self.best_prec1 = 0

    def result_init(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
        self.logger = self._get_logger(os.path.join(self.result_dir, "train_logger.log"), "train_logger")
        self.logger.info("train config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        with open(os.path.join(self.result_dir, "train_config.txt"), "w") as f:
            f.write(str(vars(self.args)))
        self.logger.info("--------- Train -----------")

    def _get_logger(self, filename, logger_name):
        import logging
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        self.logger.info(f"Loading {self.dataset_type} dataset...")
        
        transform_train = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.dataset_type == "cifar10":
            from torchvision.datasets import CIFAR10
            train_dataset = CIFAR10(root=self.dataset_dir, train=True, download=True, transform=transform_train)
            val_dataset = CIFAR10(root=self.dataset_dir, train=False, download=True, transform=transform_val)
        elif self.dataset_type == "hardfakevsrealfaces":
            csv_file = os.path.join(self.dataset_dir, "data.csv")
            root_dir = self.dataset_dir  # Points to directory with real and fake subdirs
            dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform_train)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            val_dataset.dataset.transform = transform_val
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        self.logger.info(f"{self.dataset_type} dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")
        num_classes = 2 if self.dataset_type == "hardfakevsrealfaces" else 10
    
        self.logger.info("Loading teacher model")
        if self.arch == "ResNet_50":
            self.teacher = models.resnet50(weights=None)  # Load without pretrained weights initially
            num_ftrs = self.teacher.fc.in_features
            self.teacher.fc = nn.Linear(num_ftrs, num_classes)
            if self.teacher_ckpt_path and os.path.exists(self.teacher_ckpt_path):
                ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
                self.teacher.load_state_dict(ckpt_teacher["state_dict"])
                self.logger.info("Teacher loaded from finetuned checkpoint.")
            else:
                self.logger.info("No teacher checkpoint provided. Using ImageNet pretrained weights.")
                self.teacher = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.teacher.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported architecture for teacher: {self.arch}")

        self.logger.info("Building student model")
        if self.arch == "ResNet_50":
            self.student = self._ResNet_50_sparse_imagenet(
                gumbel_start_temperature=self.gumbel_start_temperature,
                gumbel_end_temperature=self.gumbel_end_temperature,
                num_epochs=self.num_epochs,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

    class _ResNet_50_sparse_imagenet(nn.Module):
        def __init__(self, gumbel_start_temperature, gumbel_end_temperature, num_epochs, num_classes):
            super().__init__()
            self.model = models.resnet50(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.gumbel_start_temperature = gumbel_start_temperature
            self.gumbel_end_temperature = gumbel_end_temperature
            self.num_epochs = num_epochs
            self.gumbel_temperature = gumbel_start_temperature
            self.mask_modules = []
            self.ticket = False

        def forward(self, x):
            return self.model(x), []

        def update_gumbel_temperature(self, epoch):
            self.gumbel_temperature = self.gumbel_start_temperature - (self.gumbel_start_temperature - self.gumbel_end_temperature) * (epoch / self.num_epochs)

        def get_flops(self):
            return torch.tensor(Flops_baselines["ResNet_50"] * (10**6))

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss()
        self.kd_loss = self._KDLoss()
        self.rc_loss = self._RCLoss()
        self.mask_loss = self._MaskLoss()

    class _KDLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.KLDivLoss(reduction="batchmean")

        def forward(self, teacher_logits, student_logits):
            student_log_softmax = torch.log_softmax(student_logits, dim=1)
            teacher_softmax = torch.softmax(teacher_logits, dim=1)
            return self.criterion(student_log_softmax, teacher_softmax)

    class _RCLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, student_features, teacher_features):
            return torch.tensor(0.0)

    class _MaskLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, flops, flops_baseline, compress_rate):
            return torch.abs(flops - flops_baseline * compress_rate) / flops_baseline

    def define_optim(self):
        weight_params = [p for n, p in self.student.named_parameters() if p.requires_grad]
        self.optim_weight = torch.optim.Adamax(
            weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7
        )
        self.scheduler_student_weight = self._CosineAnnealingLRWarmup(
            self.optim_weight, T_max=self.lr_decay_T_max, eta_min=self.lr_decay_eta_min,
            warmup_steps=self.warmup_steps, warmup_start_lr=self.warmup_start_lr
        )

    class _CosineAnnealingLRWarmup:
        def __init__(self, optimizer, T_max, eta_min, warmup_steps, warmup_start_lr):
            self.optimizer = optimizer
            self.T_max = T_max
            self.eta_min = eta_min
            self.warmup_steps = warmup_steps
            self.warmup_start_lr = warmup_start_lr
            self.step_count = 0

        def step(self):
            self.step_count += 1
            if self.step_count <= self.warmup_steps:
                lr = self.warmup_start_lr + (self.optimizer.param_groups[0]["lr"] - self.warmup_start_lr) * (self.step_count / self.warmup_steps)
            else:
                lr = self.eta_min + 0.5 * (self.optimizer.param_groups[0]["lr"] - self.eta_min) * (1 + np.cos(np.pi * (self.step_count - self.warmup_steps) / self.T_max))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train(self):
        if self.device == "cuda":
            self.teacher = self.teacher.cuda()
            self.student = self.student.cuda()
            self.ori_loss = self.ori_loss.cuda()
            self.kd_loss = self.kd_loss.cuda()
            self.rc_loss = self.rc_loss.cuda()
            self.mask_loss = self.mask_loss.cuda()

        meter_loss = self._AverageMeter("Loss", ":.4e")
        meter_top1 = self._AverageMeter("Acc@1", ":6.2f")

        self.teacher.eval()
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.student.train()
            self.student.ticket = False
            meter_loss.reset()
            meter_top1.reset()
            lr = self.optim_weight.param_groups[0]["lr"] if epoch > 1 else self.warmup_start_lr

            self.student.update_gumbel_temperature(epoch)
            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description(f"epoch: {epoch}/{self.num_epochs}")
                for images, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits_student, feature_list_student = self.student(images)
                    with torch.no_grad():
                        logits_teacher, feature_list_teacher = self.teacher(images)
                    
                    ori_loss = self.ori_loss(logits_student, targets)
                    kd_loss = (self.target_temperature**2) * self.kd_loss(logits_teacher / self.target_temperature, logits_student / self.target_temperature)
                    rc_loss = self.rc_loss(feature_list_student, feature_list_teacher)
                    Flops_baseline = Flops_baselines.get(self.arch, 4134)
                    Flops = self.student.get_flops()
                    mask_loss = self.mask_loss(Flops, Flops_baseline * (10**6), self.compress_rate)

                    total_loss = ori_loss + self.coef_kdloss * kd_loss + self.coef_rcloss * rc_loss + self.coef_maskloss * mask_loss

                    total_loss.backward()
                    self.optim_weight.step()

                    prec1 = self._get_accuracy(logits_student, targets, topk=(1,))[0]
                    n = images.size(0)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)

                    _tqdm.set_postfix(loss=f"{meter_loss.avg:.4f}", top1=f"{meter_top1.avg:.4f}")
                    _tqdm.update(1)

            self.scheduler_student_weight.step()
            self.logger.info(f"[Train] Epoch {epoch}: LR {lr:.6f} Loss {meter_loss.avg:.4f} Prec@1 {meter_top1.avg:.2f}")

            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description(f"val epoch: {epoch}/{self.num_epochs}")
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits_student, _ = self.student(images)
                        prec1 = self._get_accuracy(logits_student, targets, topk=(1,))[0]
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)

            self.logger.info(f"[Val] Epoch {epoch}: Prec@1 {meter_top1.avg:.2f}")
            self.start_epoch += 1

        self.logger.info("Train finished!")

    class _AverageMeter:
        def __init__(self, name, fmt=":f"):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
