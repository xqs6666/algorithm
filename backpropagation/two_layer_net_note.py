import numpy as np
from collections import OrderedDict
from torchvision import datasets
import torch 


# -----------------------------
# 仔细阅读：本文件所示模型
# - 两层全连接神经网络（Input -> Affine1 -> ReLU -> Affine2 -> Softmax+CrossEntropy）
# - 使用 numpy 实现前向/反向传播（手写小型神经网络实现）
# - 训练数据为 MNIST（28x28 灰度手写数字）
# 关键知识点：
# - 参数共享 vs 变量重绑定（为什么要使用 `params[...] = ...` / `params -= ...`）
# - softmax 数值稳定化（减去每行最大值）
# - one-hot 与交叉熵的结合，损失对 logits 的梯度为 (y - t) / N
# -----------------------------


class Affine:
    """
    全连接（仿射）层（y = xW + b）
    - W: (in_dim, out_dim)
    - b: (out_dim,)
    保存的中间量：
    - self.x: 前向输入（用于计算 dW）
    - self.dW, self.db: 在 backward 中计算并保存梯度
    注意事项：
    - 这里我们假设 W, b 是 numpy 数组且可能与外部 params 字典共享同一块内存。
      因此训练更新时应尽量在原地修改 params（如 params[key] -= lr * grad），
      以保证 Affine 实例中保存的 W/b 引用能感知修改。
    """
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        # x: (N, in_dim)
        # 输出: (N, out_dim)
        self.x = x
        out = np.dot(x, self.W) + self.b  # 广播：b 加到每个样本
        return out
    
    def backward(self,dout):
        """
        dout: (N, out_dim) — 上游传来的对输出的梯度
        目标：计算本层对输入 x 的梯度 dx，
             并计算本层参数 W,b 的梯度 dW, db
        推导：
        - y = x W + b
        - dW = x^T dout
        - db = sum(dout over batch axis)
        - dx = dout W^T
        """
        dx = np.dot(dout, self.W.T)             # (N, in_dim)
        dW = np.dot(self.x.T, dout)             # (in_dim, out_dim)
        db = np.sum(dout, axis=0)               # (out_dim,)
        self.dW = dW
        self.db = db
        return dx


class SoftmaxWithLoss:
    """
    Softmax + CrossEntropy 一体化层（常见实现）
    - forward(logits, t_onehot) -> 返回 batch 平均交叉熵损失
    - backward() -> 返回 logits 的梯度 (y - t) / N
    说明：
    - 使用 one-hot 标签 t（shape (N, C)）
    - softmax 使用数值稳定化：exp(x - max(x))
    - cross_entropy_error 对 one-hot 标签的实现为 -sum(t * log(y)) / N
    """
    def __init__(self):
        self.loss = None
        self.y = None   # softmax 输出概率
        self.t = None   # one-hot 标签

    def softmax(self,x):
        # x: (N, C)
        # 为了数值稳定性，每行减去最大值
        np_max = np.max(x, axis=1, keepdims=True)    # (N,1)
        np_x = np.exp(x - np_max)
        return np_x / np.sum(np_x, axis=1, keepdims=True)

    def cross_entropy_error(self,y,t):
        # y: (N,C) 概率； t: (N,C) one-hot
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        # 加上一个很小的数避免 log(0)
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
        
    def forward(self,x,t):
        # x: logits (N,C); t: one-hot (N,C)
        y = self.softmax(x)
        loss = self.cross_entropy_error(y, t)
        self.t = t
        self.y = y
        self.loss = loss
        return loss
    
    def backward(self, dout=1):
        # dout 默认为 1（dLoss / dLoss）
        batch_size = self.t.shape[0]
        # 对 logits 的梯度 (常用结论)：(y - t) / N
        dx = (self.y - self.t) * dout / batch_size
        return dx
    

class Relu:
    """
    ReLU 激活层（非线性）
    - forward: 保存 mask，用于 backward（把输入中 <= 0 的位置置为 0）
    - backward: 上游梯度中对应 mask 的位置置为 0
    注意：这里使用了原地修改上游的 dout（dout[self.mask] = 0） ——
    在某些场景下为了安全可以先复制 dout，再修改；但为了效率我们直接在传入的 dout 上改。
    """
    def __init__(self):
        self.mask = None

    def forward(self,x):
        # mask 标记哪里 <= 0，表示这些位置在 backward 时梯度为 0
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        # dout 必须和 forward 输入形状一致
        dout[self.mask] = 0   # 把不可导的地方的梯度置 0
        dx = dout
        return dx
    

class TwoLayerNet:
    """
    两层网络的容器：
    - params: 保存参数的字典（W1,b1,W2,b2）
    - layers: 保存网络层的 OrderedDict（前向顺序）
    - lastLayer: SoftmaxWithLoss 实例（loss 层）
    重点：Affine 层构造时我们直接传入了 self.params[...] 的引用，
    因此如果外部代码以原地方式修改 self.params[...]（例如 self.params[key] -= ...），
    Affine 层内部看到的 W/b 会随之更新。若使用重新绑定（A = A - B），
    Affine 内部引用不会改变，从而导致训练失败（参数看起来没有更新）。
    """
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        # 权重初始化（高斯，乘以较小 std），偏置初始化为 0
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # 构建层（顺序很重要）
        self.layers = OrderedDict()
        # 注意：Affine 接收的是 numpy 数组引用
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        """ 前向计算（只到 logits） """
        data = x
        for layer in self.layers.values():
            data = layer.forward(data)
        return data

    def loss(self,x,t):
        """ 计算损失（利用 lastLayer） """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self,x,t):
        """ 计算识别精度（注意 t 可能是 one-hot 或者整数标签） """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = (np.sum(y == t) / float(x.shape[0]))
        return accuracy

    def gradient(self,x,t):
        """
        反向传播求梯度：
        - 先 forward 计算 loss（并在 lastLayer 保存 y, t）
        - lastLayer.backward 得到 logits 的梯度
        - 依次对倒序 layers 调用 backward，计算各层梯度并保存
        - 最后从 Affine 层读取 dW, db 返回
        """
        self.loss(x,t)

        # backward
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

class SGD:
    def __init__(self,learning=0.01):
        self.learning = learning
        pass

    def update(self,params,grads):
        for key in params.keys():
            params[key] -=  self.learning*grads[key]

class Momentum:
    def __init__(self,momentum=0.9,learning=0.01):
        self.v = {}
        self.momentum = momentum
        self.learning= learning
        pass

    def update(self,parames,grad):
        if self.v == {}:
            for key,val in parames.items():
                self.v[key] = np.zeros_like(val)
        
        for key,val in parames.items():
            self.v[key] = self.momentum * self.v[key] - self.learning*grad[key]
            parames[key] +=  self.v[key]

def build_data():
    """
    加载 MNIST 数据并做预处理：
    - 归一化到 [0,1]
    - 展平为 (N, 28*28)
    - 把标签转换为 one-hot（返回 numpy 数组）
    注意：这里使用 torchvision.datasets.MNIST，返回的 dataset.data 是 uint8
    """
    global x_train
    global t_train_onehot
    global x_test
    global t_test_onehot

    dataset_train = datasets.MNIST(root='./data', download=True, train=True)
    x_train, t_train = dataset_train.data.numpy(), dataset_train.targets.numpy()
    x_train = x_train / 255.0  # 归一化
    x_train = x_train.reshape(60000, 28*28)

    dataset_test = datasets.MNIST(root='./data', download=True, train=False)
    x_test, t_test = dataset_test.data.numpy(), dataset_test.targets.numpy()
    x_test = x_test.reshape(10000, 28*28)
    x_test = x_test / 255.0  # 归一化

    # 使用 torch 的 one_hot 工具然后转回 numpy（仅为了方便）
    t_train_onehot = torch.nn.functional.one_hot(torch.tensor(t_train), num_classes=10).numpy()
    t_test_onehot = torch.nn.functional.one_hot(torch.tensor(t_test), num_classes=10).numpy()
    return (x_train, t_train_onehot), (x_test, t_test_onehot)


# ------- 主流程 -------
(x_train, t_train_onehot), (x_test, t_test_onehot) = build_data()

network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 判断是否到达一个完整的 epoch（轮次）
# 建议：使用整数除法，保证 iter_per_epoch 为整数
# 你原来的写法 iter_per_epoch = max(train_size / batch_size, 1) 会得到浮点数，
# 在后面用 i % iter_per_epoch 时可能产生意想不到的结果（虽然 Python 支持 float % int，
# 但直观上我们希望是整数）。
iter_per_epoch = max(train_size // batch_size, 1) #600 

# optimizer = SGD()
optimizer = Momentum()

for i in range(iters_num):
    # 随机采样一个迷你批
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train_onehot[batch_mask]

    # 计算梯度（反向传播）
    grad = network.gradient(x_batch, t_batch)

    # 关键：原地更新参数（这样 Affine 层内部引用到的参数内存会被修改）
    # 以下写法会在原地修改 numpy 数组（等同于 params[key] -= ...）
    # 这样 Affine 层中保存的 W/b 引用会看到更新后的值，从而训练能够正常进行。
    # for key in ("W1", "b1", "W2", "b2"):
    #     # 原地更新（等价于 network.params[key] = network.params[key] - lr * grad[key]
    #     # 但使用 '-=' 会尽量避免产生新的数组）
    #     network.params[key] -= learning_rate * grad[key]
    
    
    optimizer.update(network.params,grad)


    # 记录并打印损失 / 精度
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每个 epoch 打印一次（注意 iter_per_epoch 是整数）
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train_onehot)
        test_acc = network.accuracy(x_test, t_test_onehot)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"iter {i}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
