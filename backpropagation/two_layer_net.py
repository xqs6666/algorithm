import numpy as np
from collections import OrderedDict
from torchvision import datasets
import torch 

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        #x (N,H)
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    
    def backward(self,dout):
        # 计算对输入的梯度
        dx = np.dot(dout,self.W.T)
        # 计算对权重的梯度
        dW = np.dot(self.x.T,dout)
        # 计算对偏置的梯度 - 对batch维度求和
        db = np.sum(dout,axis=0)
        self.dW = dW
        self.db = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def softmax(self,x):
        np_max = np.max(x,axis=1,keepdims=True)
        np_x = np.exp(x-np_max)
        return np_x/np.sum(np_x,axis=1,keepdims=True)
    
    def cross_entropy_error(self,y,t):
        if y.ndim ==1:
            y = y.reshape(1,y.size)
            t = t.reshape(1,t.size)
        batch_size = y.shape[0]
        return -np.sum(t*np.log(y+1e-7))/batch_size
        
    def forward(self,x,t):
        y = self.softmax(x)
        loss = self.cross_entropy_error(y,t)
        self.t = t
        self.y = y
        self.loss = loss
        return loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx
    
class Relu:
    def __init__(self):
        self.mask = None
        pass

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    """
    反向传播
    dout: 上游传来的梯度
    """    
    def backward(self,dout):
        # dout 必须与 forward 的输出形状相同
        # 也就是与输入x的形状相同
        dout[self.mask] = 0
        dx = dout
        return dx
    
       
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"],self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"],self.params["b2"])
        
        self.lastLayer = SoftmaxWithLoss()
        pass

    def predict(self,x):
        data = x
        for layer in self.layers.values():
            data = layer.forward(data)
        return data

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    '''
    计算识别精度
    '''
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:
            t = np.argmax(t,axis=1)
        accuracy = (np.sum(y == t)/float(x.shape[0]))
        return accuracy


    def gradient(self,x,t):
        self.loss(x,t)

        #backward
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
    global x_train
    global t_train_onehot
    global x_test
    global t_test_onehot

    dataset_train = datasets.MNIST(root='./data', download=True,train=True)
    x_train, t_train = dataset_train.data.numpy(), dataset_train.targets.numpy()
    x_train = x_train / 255.0  # 添加归一化
    x_train = x_train.reshape(60000,28*28)

    dataset_train = datasets.MNIST(root='./data', download=True,train=False)
    x_test, t_test = dataset_train.data.numpy(), dataset_train.targets.numpy()
    x_test = x_test.reshape(10000,28*28)
    x_test = x_test / 255.0  # 添加归一化

    t_train_onehot = torch.nn.functional.one_hot(torch.tensor(t_train), num_classes=10).numpy()
    t_test_onehot = torch.nn.functional.one_hot(torch.tensor(t_test), num_classes=10).numpy()
    return (x_train,t_train_onehot),(x_test,t_test_onehot)


(x_train,t_train_onehot),(x_test,t_test_onehot) = build_data()


network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train_onehot[batch_mask]

    grad = network.gradient(x_batch,t_batch)

    for key in ("W1","b1","W2","b2"):
        network.params[key] -= learning_rate*grad[key]


    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i%iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train_onehot)
        test_acc = network.accuracy(x_test, t_test_onehot)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)

