from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

class DigitalModel:
    def __init__(self):
        self.network = self.init_network()  # ✅ 初始化一次，多次使用
    
    def get_data(self):
        dataset_train = datasets.MNIST(root='./data', download=True, train=True)
        x_train = dataset_train.data.numpy().astype(np.float32)
        x_label = dataset_train.targets.numpy()
        
        # ✅ 数据归一化到 [0, 1]
        x_train = x_train.reshape(60000, 28*28) / 255.0
        return x_train, x_label
    
    def sigmoid(self, x):
        # ✅ 数值稳定性处理
        x = np.clip(x, -500, 500)  # 防止指数溢出
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        # ✅ 你的实现已经是正确的
        c = np.max(x, axis=-1, keepdims=True)
        exp_a = np.exp(x - c)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)
    
    def init_network(self):
        network = {}
        # ✅ 改进的权重初始化（Xavier/Glorot初始化）
        network["W1"] = np.random.randn(784, 50) * np.sqrt(1.0 / 784)
        network["b1"] = np.zeros(50)  # ✅ 偏置初始为0
        
        network["W2"] = np.random.randn(50, 100) * np.sqrt(1.0 / 50)
        network["b2"] = np.zeros(100)
        
        network["W3"] = np.random.randn(100, 10) * np.sqrt(1.0 / 100)
        network["b3"] = np.zeros(10)
        return network
    
    def predict(self, x):
        # ✅ 使用已经初始化好的网络权重
        W1, W2, W3 = self.network["W1"], self.network["W2"], self.network["W3"]
        b1, b2, b3 = self.network["b1"], self.network["b2"], self.network["b3"]
        
        # 前向传播
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        return y

# 测试修复后的代码
dm = DigitalModel()
x_train, x_label = dm.get_data()

# accuracy_count = 0
# total_tests = 1000  # 测试少量样本看看效果

# for idx in range(total_tests):
#     data = x_train[idx]
#     y = dm.predict(data)
#     p = np.argmax(y)
#     if p == x_label[idx]:
#         accuracy_count += 1

# accuracy = float(accuracy_count) / total_tests
# print(f"测试样本数: {total_tests}")
# print(f"正确预测数: {accuracy_count}")
# print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

batch_size = 100
accuracy_count = 0
for i in range(0,len(x_train),batch_size):
    data = x_train[i:i+batch_size]
    y = dm.predict(data)
    p = np.argmax(y,axis=1)
    accuracy_count = accuracy_count+np.sum(p==x_label[i:i+batch_size])

accuracy = float(accuracy_count) / len(x_train)
print(f"测试样本数: {len(x_train)}")
print(f"正确预测数: {accuracy_count}")
print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")