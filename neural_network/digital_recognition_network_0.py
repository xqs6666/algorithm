import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.save("./images/mnist_digit.png")


dataset_train = datasets.MNIST(root='./data', download=True,train=True)
dataset_test = datasets.MNIST(root='./data', download=True,train=False)

x_train, x_label = dataset_train.data.numpy(), dataset_train.targets.numpy() 
t_test, t_label = dataset_test.data.numpy(), dataset_test.targets.numpy()
print(x_train.shape, x_label.shape)  # 输出: (60000, 28, 28) (60000,)
print(t_test.shape, t_label.shape)    # 输出: (10000, 28, 28) (10000,)

# 将数据从 (60000, 28, 28) 转换为 (60000, 784)
x_train=x_train.reshape(60000, 28*28)
# 将数据从 (10000, 28, 28) 转换为 (10000, 784)
t_test=t_test.reshape(10000, 28*28)

print(x_label.shape, t_label.shape)  # 输出: (60000,) (10000,)

img = x_train[0]
label = x_label[0]

img = img.reshape(28,28)
img_show(img)

class DigitalModel:
    def __init__(self):
        pass

    def get_data(self):
        dataset_train = datasets.MNIST(root='./data', download=True,train=True)
        x_train, x_label = dataset_train.data.numpy(), dataset_train.targets.numpy()
        x_train = x_train.reshape(60000,28*28) 
        return x_train, x_label

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def softmax(self,x):
        c = np.max(x)
        exp_a_n = np.exp(x-c)
        exp_a_sum = np.sum(exp_a_n)
        return exp_a_n/exp_a_sum
    
    def init_network(self):
        network={}
        network["W1"] = np.random.randn(784,50)
        network["b1"] = np.ones(50)
        network["W2"] = np.random.randn(50,100)
        network["b2"] = np.ones(100)
        network["W3"] = np.random.randn(100,10)
        network["b3"] = np.ones(10)
        return network

    def predict(self,x):
        network = self.init_network()
        W1, W2, W3 = network["W1"], network["W2"], network["W3"]
        b1, b2, b3 = network["b1"], network["b2"], network["b3"]

        a1 = np.dot(x,W1)+b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1,W2)+b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2,W3)+b3
        y = self.softmax(a3)
        return y

# 测试修复后的代码
dm = DigitalModel()
x_train, x_label = dm.get_data()

accuracy_count = 0
total_tests = 1000  # 测试少量样本看看效果

for idx in range(total_tests):
    data = x_train[idx]
    y = dm.predict(data)
    p = np.argmax(y)
    if p == x_label[idx]:
        accuracy_count += 1

accuracy = float(accuracy_count) / total_tests
print(f"测试样本数: {total_tests}")
print(f"正确预测数: {accuracy_count}")
print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")