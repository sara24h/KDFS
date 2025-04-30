import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from .layer import SoftMaskedConv2d

class MaskedNet(nn.Module):
    def __init__(self, gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=3):
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
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        Flops_total = torch.tensor(0.0)
        Flops_total += 112 * 112 * 7 * 7 * 3 * 64 + 112 * 112 * 64
        for i, m in enumerate(self.mask_modules):
            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0
            if len(self.mask_modules) == 48:
                if i % 3 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 3 == 2:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 * (m.out_channels // 4) * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels
            elif len(self.mask_modules) in [16, 32]:
                if i % 2 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 2 == 1 and i != 1:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 * m.out_channels * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels

            Flops_total += Flops_conv + Flops_bn + Flops_shortcut_conv + Flops_shortcut_bn
        return Flops_total

class BasicBlock_sparse(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, ticket, gumbel_temperature=None):
        out = F.relu(self.bn1(self.conv1(x, ticket, gumbel_temperature=gumbel_temperature)))
        out = self.bn2(self.conv2(out, ticket, gumbel_temperature=gumbel_temperature))
        out += self.downsample(x)
        out = F.relu(out)
        return out

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
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, ticket, gumbel_temperature=None):
        out = F.relu(self.bn1(self.conv1(x, ticket, gumbel_temperature=gumbel_temperature)))
        out = F.relu(self.bn2(self.conv2(out, ticket, gumbel_temperature=gumbel_temperature)))
        out = self.bn3(self.conv3(out, ticket, gumbel_temperature=gumbel_temperature))
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
        num_epochs=3,
    ):
        super().__init__(
            gumbel_start_temperature,
            gumbel_end_temperature,
            num_epochs,
        )
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        expansion = block.expansion
        self.shortcut1 = nn.Conv2d(64 * expansion, 64 * expansion, kernel_size=1)
        self.shortcut2 = nn.Conv2d(128 * expansion, 128 * expansion, kernel_size=1)
        self.shortcut3 = nn.Conv2d(256 * expansion, 256 * expansion, kernel_size=1)
        self.shortcut4 = nn.Conv2d(512 * expansion, 512 * expansion, kernel_size=1)

        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, gumbel_temperature=None):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out, self.ticket, gumbel_temperature=gumbel_temperature)
        feature_list.append(self.shortcut1(out))

        for block in self.layer2:
            out = block(out, self.ticket, gumbel_temperature=gumbel_temperature)
        feature_list.append(self.shortcut2(out))

        for block in self.layer3:
            out = block(out, self.ticket, gumbel_temperature=gumbel_temperature)
        feature_list.append(self.shortcut3(out))

        for block in self.layer4:
            out = block(out, self.ticket, gumbel_temperature=gumbel_temperature)
        feature_list.append(self.shortcut4(out))

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_18_sparse_hardfakevsreal(
    gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=3
):
    return ResNet_sparse(
        block=BasicBlock_sparse,
        num_blocks=[2, 2, 2, 2],
        num_classes=2,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )

def ResNet_34_sparse_hardfakevsreal(
    gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=3
):
    return ResNet_sparse(
        block=BasicBlock_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=2,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )

def ResNet_50_sparse_hardfakevsreal(
    gumbel_start_temperature=1.0, gumbel_end_temperature=0.1, num_epochs=3
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=2,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )
