import numpy as np

import numpy as np

def batch_norm(x, gamma=1, beta=0, eps=1e-5):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out

# 模拟一个输入层：100 个样本，每个 5 维特征
x = np.random.randn(100, 5) * 10 + 50  # 均值大，方差大
x_bn = batch_norm(x)

print("原始输入：均值 =", np.mean(x, axis=0))
print("BatchNorm后：均值 =", np.mean(x_bn, axis=0))
print("BatchNorm后：方差 =", np.var(x_bn, axis=0))
