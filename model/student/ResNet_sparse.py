"""
Sparse ResNet50 in PyTorch for binary classification (hardfakevsreal)

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from .layer import SoftMaskedConv2d


class MaskedNet(nn.Module):
    def __init__(
        self, gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=100
    ):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []

    def checkpoint(self):
        for m in self.mask_modules:
            m.checkpoint()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = max(
            self.gumbel_end_temperature,
            self.gumbel_start_temperature * (
                self.gumbel_end_temperature / self.gumbel_start_temperature
            ) ** (epoch / self.num_epochs)
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        Flops_total = torch.tensor(0.0, device=next(self.parameters()).device)
        # Input size: 300x300, after conv1 (7x7, stride=2): 150x150, after maxpool (3x3, stride=2): 75x75
        Flops_total += 150 * 150 * 7 * 7 * 3 * 64 + 150 * 150 * 64  # conv1 + bn1
        feature_sizes = [
            (75, 75),  # layer1
            (38, 38),  # layer2 (after stride=2)
            (19, 19),  # layer3 (after stride=2)
            (10, 10)   # layer4 (after stride=2)
        ]
        channels = [256, 512, 1024, 2048]  # Output channels after each layer (expansion=4)
        for i, m in enumerate(self.mask_modules):
            layer_idx = i // 9  # 3 convs per bottleneck, 3+4+6+3 bottlenecks
            h, w = feature_sizes[layer_idx]
            if i % 9 == 0:  # conv1 (1x1)
                in_ch = channels[layer_idx-1] if i > 0 else 64
                out_ch = m.mask.sum()
                Flops_conv = h * w * 1 * 1 * in_ch * out_ch
            elif i % 9 == 1:  # conv2 (3x3)
                in_ch = self.mask_modules[i-1].mask.sum()
                out_ch = m.mask.sum()
                Flops_conv = h * w * 3 * 3 * in_ch * out_ch
            else:  # conv3 (1x1)
                in_ch = self.mask_modules[i-1].mask.sum()
                out_ch = m.mask.sum()
                Flops_conv = h * w * 1 * 1 * in_ch * out_ch
            Flops_bn = h * w * m.mask.sum()
            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0
            if i % 9 == 8:  # Downsample for first block in layer2, layer3, layer4
                if i in [9, 27, 63]:  # First bottleneck in layer2, layer3, layer4
                    Flops_shortcut_conv = h * w * 1 * 1 * channels[layer_idx-1] * channels[layer_idx]
                    Flops_shortcut_bn = h * w * channels[layer_idx]
            Flops_total += Flops_conv + Flops_bn + Flops_shortcut_conv + Flops_shortcut_bn
        return Flops_total


class Bottleneck_sparse(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SoftMaskedConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
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
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = F.relu(self.bn2(self.conv2(out, ticket)))
        out = self.bn3(self.conv3(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet_sparse(MaskedNet):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=2,
        gumbel_start_temperature=1.0,
        gumbel_end_temperature=0.1,
        num_epochs=100
    ):
        super().__init__(
            gumbel_start_temperature,
            gumbel_end_temperature,
            num_epochs
        )
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        # Remove feature conversion layers since feature shapes are compatible
        # self.convert1, self.convert2, etc., are not needed for same architecture

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out, self.ticket)
        feature_list.append(out)
        out = self.layer2(out, self.ticket)
        feature_list.append(out)
        out = self.layer3(out, self.ticket)
        feature_list.append(out)
        out = self.layer4(out, self.ticket)
        feature_list.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list


def ResNet_50_sparse_imagenet(
    gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=100
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=2,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs
    )
