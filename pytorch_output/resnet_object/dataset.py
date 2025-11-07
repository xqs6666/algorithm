"""
dataset.py
数据加载模块：封装 Dataset 和 DataLoader，便于替换数据集或增强策略。
当前以 CIFAR-10 为示例（RGB, 10 类）。你可以替换为 ImageFolder / 自定义 Dataset。
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

def get_transforms(train=True):
    """
    返回训练或测试时使用的数据增强 / 预处理流水线。
    - 训练：包含随机裁剪、水平翻转等增强提高泛化。
    - 测试：保持一致性（Resize -> CenterCrop -> ToTensor -> Normalize）。
    注意：ResNet 原始设计用于 ImageNet（224x224），所以这里将 CIFAR10 调整到 224。
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪并缩放到 224（增强）
            transforms.RandomHorizontalFlip(),  # 随机翻转
            transforms.ToTensor(),
            # Normalize 可加上 ImageNet 的均值方差或 CIFAR 自己的均值方差
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, dataset_root=DATA_DIR):
    """
    构建并返回 train_loader, test_loader。
    - dataset_root: 数据存放目录
    - 使用 torchvision.datasets.CIFAR10 做示例；换数据集时可替换此处
    """
    os.makedirs(dataset_root, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=dataset_root, train=True,
                                     transform=get_transforms(train=True), download=True)
    test_dataset = datasets.CIFAR10(root=dataset_root, train=False,
                                    transform=get_transforms(train=False), download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=PIN_MEMORY)
    return train_loader, test_loader
