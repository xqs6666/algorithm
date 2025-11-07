"""
model.py
ResNet 模型定义（支持 BasicBlock 和 resnet18 配置）。
包含：
- BasicBlock（用于 ResNet-18/34）
- ResNet 主体（可通过 layers 参数切换层数）
- 工厂函数 resnet18()
注释写得详细，方便理解每一步为何设计成这样。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic residual block（ResNet-18/34 使用）。
    结构：
      conv3x3 -> BN -> ReLU -> conv3x3 -> BN
      shortcut: identity（或 1x1 conv + BN 用于维度匹配 / 下采样）
    残差相加后再做 ReLU。
    """
    expansion = 1  # BasicBlock 不改变通道数（扩展倍数为1）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 模块内部的输出通道数（block 的主路径输出）
            stride: 第一层 conv 的 stride（用于下采样）
            downsample: 若非 None，则用于调整 identity 的尺寸（通常是 1x1 conv + BN）
        """
        super().__init__()
        # 第一个 3x3 卷积，可能伴随 stride（实现下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个 3x3 卷积，stride 固定为1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果需要调整 identity（例如通道变化或步长不为1），使用 downsample
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保存 shortcut 分支的输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果 downsample 存在（例如通道数不同或 stride>1），调整 identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差相加
        out = self.relu(out)  # 再激活
        return out


class ResNet(nn.Module):
    """
    ResNet 主体。通过传入 block、layers 列表来定义不同的网络深度。
    - block: 残差块类型（BasicBlock 或 Bottleneck）
    - layers: 每个 stage 的 block 数量（如 [2,2,2,2] 表示 ResNet-18）
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64  # 初始通道数（stem 输出）

        # ---------- stem: 首个大卷积、BN、ReLU、MaxPool ----------
        # 与原始 ResNet 保持一致：7x7 conv stride=2，随后 3x3 maxpool stride=2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 将输入从 224x224 下采样为 56x56（224 -> /2 -> /2）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---------- 四个 stage（每个 stage 由多个 block 组成） ----------
        # 每个 stage 通道数分别为 64, 128, 256, 512
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 无论输入大小如何，输出 1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化（使用与论文一致的做法）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态初始化（适用于 ReLU）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BN 的 gamma 初始化为 1，beta 初始化为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 可选：将 residual 分支的最后一个 BN 的 gamma 初始化为 0（有利于收敛）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建一个 stage（由 blocks 个残差块组成）。
        第一块可能因为 stride != 1 或通道数不同而需要 downsample。
        """
        downsample = None
        # 当 stride != 1（需要下采样）或者输入通道 != 输出通道*expansion（需要调整通道数）时，构造 downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 使用 1x1 卷积调整通道与步长，随后 BN
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一块负责可能的下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 更新 in_channels（供后续块使用）
        self.in_channels = out_channels * block.expansion
        # 后续 blocks 个基本块（stride=1）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个 stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 池化与分类
        x = self.avgpool(x)               # B x C x 1 x 1
        x = torch.flatten(x, 1)           # B x C
        x = self.fc(x)                    # B x num_classes
        return x


def resnet18(num_classes=1000, **kwargs):
    """构造 ResNet-18 的工厂函数"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


# 若需要，可以在底部做一个简单的 smoke-test（仅作本地测试）
if __name__ == "__main__":
    model = resnet18(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # 期望 (2, 10)
