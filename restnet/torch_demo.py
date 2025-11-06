import torch.nn as nn
import torch

# 定义这个卷积层
channels = 256
refinement_conv = nn.Conv2d(channels, channels, 3, padding=1)
'''
输入32通道 输出32通道 卷积核3x3 padding=1 =====> 输出特征图32个 大小不变
(32,255,254) * (32,32,3,3)|padding=1| ==>(32,255,254)
'''
input_features = torch.randn(32,256,64,64)
print(f"输入特征图形状: {input_features.shape}")

output_features = refinement_conv(input_features)
print(f"输出特征图形状: {output_features.shape}")