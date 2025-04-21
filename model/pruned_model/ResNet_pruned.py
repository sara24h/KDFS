"""
Pruned ResNet50 in PyTorch for binary classification (hardfakevsreal)

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_preserved_filter_num(mask):
    return int(mask.sum())


class Bottleneck_pruned(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks
        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes, preserved_filter_num1, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)
        preserved_filter_num3 = get_preserved_filter_num(masks[2])
        self.conv3 = nn.Conv2d(
            preserved_filter_num2,
            preserved_filter_num3,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(preserved_filter_num3)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut_out = self.downsample(x)
        # Ensure output channels match shortcut by selecting preserved filters
        if out.size(1) != shortcut_out.size(1):
            padded_out = torch.zeros_like(shortcut_out)
            preserved_indices = torch.where(self.masks[2] == 1)[0]
            padded_out[:, preserved_indices, :, :] = out
            out = padded_out
        out += shortcut_out
        out = F.relu(out)
        return out


class ResNet_pruned(nn.Module):
    def __init__(self, block, num_blocks, masks=[], num_classes=2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=1, masks=masks[0:9]
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, masks=masks[9:21]
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, masks=masks[21:39]
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, masks=masks[39:48]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, masks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        mask_idx = 0
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    masks[mask_idx:mask_idx + 3],
                    stride
                )
            )
            self.in_planes = planes * block.expansion
            mask_idx += 3
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        feature_list.append(out)
        out = self.layer2(out)
        feature_list.append(out)
        out = self.layer3(out)
        feature_list.append(out)
        out = self.layer4(out)
        feature_list.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list


def ResNet_50_pruned_imagenet(masks):
    return ResNet_pruned(
        block=Bottleneck_pruned, num_blocks=[3, 4, 6, 3], masks=masks, num_classes=2
    )
