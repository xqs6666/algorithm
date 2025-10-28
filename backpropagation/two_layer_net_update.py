from collections import OrderedDict
import numpy as np
from torchvision import datasets
import torch

# 先定义所有层类
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        np_max = np.max(x, axis=1, keepdims=True)
        np_x = np.exp(x - np_max)
        return np_x / np.sum(np_x, axis=1, keepdims=True)
    
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
        
    def forward(self, x, t):
        self.y = self.softmax(x)
        self.t = t
        self.loss = self.cross_entropy_error(self.y, t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        data = x
        for layer in self.layers.values():
            data = layer.forward(data)
        return data

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(t.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 前向传播计算损失
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        
        return grads

def build_data():
    # 训练数据
    dataset_train = datasets.MNIST(root='./data', download=True, train=True)
    x_train, t_train = dataset_train.data.numpy(), dataset_train.targets.numpy()
    x_train = x_train.reshape(60000, 28*28).astype(np.float32) / 255.0  # 先reshape再归一化
    
    # 测试数据 - 修正变量名
    dataset_test = datasets.MNIST(root='./data', download=True, train=False)
    x_test, t_test = dataset_test.data.numpy(), dataset_test.targets.numpy()
    x_test = x_test.reshape(10000, 28*28).astype(np.float32) / 255.0
    
    # one-hot编码
    t_train_onehot = torch.nn.functional.one_hot(torch.tensor(t_train), num_classes=10).numpy()
    t_test_onehot = torch.nn.functional.one_hot(torch.tensor(t_test), num_classes=10).numpy()
    
    return (x_train, t_train_onehot), (x_test, t_test_onehot)

# 主程序
if __name__ == "__main__":
    (x_train, t_train_onehot), (x_test, t_test_onehot) = build_data()
    
    print(f"数据形状: x_train {x_train.shape}, t_train {t_train_onehot.shape}")
    print(f"数据范围: x_train [{x_train.min():.3f}, {x_train.max():.3f}]")
    
    network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1  # 可以先用0.1试试
    
    iter_per_epoch = max(train_size // batch_size, 1)
    
    print("开始训练...")
    for i in range(iters_num):
        #从0到59999中随机选择100个不重复的索引
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask] #(100,784)
        t_batch = t_train_onehot[batch_mask] #(100,784)
        
        # 梯度计算和参数更新
        grad = network.gradient(x_batch, t_batch)
        
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]
        
        # 每epoch输出一次
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_batch, t_batch)  # 先用batch数据测试
            test_acc = network.accuracy(x_test[:1000], t_test_onehot[:1000])  # 用部分测试数据
            
            loss = network.loss(x_batch, t_batch)
            epoch = int(i / iter_per_epoch)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
            
            # 如果准确率没有提升，可以调整学习率
            if epoch > 5 and train_acc < 0.2:
                learning_rate *= 0.5
                print(f"降低学习率到: {learning_rate}")
            
        if train_acc>0.98 and test_acc>0.98:
            np.savez("two_layer_net_params.npz",
            W1=network.params["W1"],
            b1=network.params["b1"],
            W2=network.params["W2"],
            b2=network.params["b2"])