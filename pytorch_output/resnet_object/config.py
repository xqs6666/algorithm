"""
config.py
配置文件：集中保存超参数与路径，便于工程化管理与实验复现。
修改这里的值即可控制训练/推理行为，不需要到处改代码。
"""

import os

# ========== 通用配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 项目根目录（当前文件夹）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")             # 数据目录（可改为实际路径）
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")# 模型保存目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== 训练超参数 ==========
NUM_CLASSES = 10            # 分类数量（CIFAR-10 示例）。更换数据集时请修改。
BATCH_SIZE = 64             # 训练/测试批量大小
NUM_WORKERS = 4             # DataLoader 的 num_workers，Linux 推荐 >0
NUM_EPOCHS = 30             # 训练轮数
LR = 1e-3                   # 初始学习率
WEIGHT_DECAY = 1e-4         # 权重衰减（L2 正则化）
PIN_MEMORY = True           # DataLoader pin_memory（若使用 CUDA 推荐 True）

# ========== 设备配置 ==========
DEVICE = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES", "") != "" or __import__("torch").cuda.is_available()) else "cpu"

# ========== 日志 / 保存 ==========
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet.pth")   # 最佳模型
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_resnet.pth")   # 最近一次训练的模型
SAVE_EPOCH_INTERVAL = 1   # 每多少个 epoch 保存一次

# ========== 其它 ==========
SEED = 42                 # 随机种子（便于复现）