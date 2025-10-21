import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader

class TwoLayerNet:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.param={}
        self.param["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.param["b1"] = np.zeros(hidden_size)

        self.param["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.param["b2"] = np.zeros(output_size)

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    
    def softmax(self,x):
        if x.ndim==2:
            c = np.max(x, axis=1, keepdims=True)
            exp_a_n = np.exp(x-c)
            exp_a_sum = np.sum(exp_a_n)
            return exp_a_n/exp_a_sum
        c = np.max(x)
        exp_a_n = np.exp(x-c)
        exp_a_sum = np.sum(exp_a_n)
        return exp_a_n/exp_a_sum
    
    def cross_entropy_error(self,y,t):
        if y.ndim == 1:
            t = t.reshape(1,t.size)
            y = y.reshape(1,y.size)
        batch_size = y.shape[0]
        correct_probs = y[np.arange(batch_size),t]#获取y_pred每一行对应t位置的正确概率是多少
        return -np.sum(np.log(correct_probs+1e-7)) / batch_size
    
    def __numerical_gradient(self,f,x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x) # 生成和x形状相同的数组

        it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]

            # f(x+h)的计算
            x[idx] = tmp_val + h
            fxh1 = f(x)

            # f(x-h)的计算
            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmp_val # 还原值

            it.iternext()
        return grad

    def predict(self,x):
        W1, b1 = self.param["W1"], self.param["b1"]
        W2, b2 = self.param["W2"], self.param["b2"]

        v1 = np.dot(x,W1)+b1
        y1 = self.sigmoid(v1)

        v2 = np.dot(y1,W2)+b2
        y2 = self.sigmoid(v2)

        y = self.softmax(y2)
        return y
    

    def loss(self,x,t):  
        y = self.predict(x)
        return self.cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x=x)
        y = np.argmax(y,axis=1)
        
        accuracy = np.sum(y==t) / float(y.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = self.__numerical_gradient(loss_W, self.param['W1'])
        grads['b1'] = self.__numerical_gradient(loss_W, self.param['b1'])
        grads['W2'] = self.__numerical_gradient(loss_W, self.param['W2'])
        grads['b2'] = self.__numerical_gradient(loss_W, self.param['b2'])
        return grads

dataset_train_0 = datasets.MNIST(root='./data', download=True,train=True)
x_train, t_train = dataset_train_0.data.numpy(), dataset_train_0.targets.numpy()
x_train = x_train.reshape(60000,28*28)

dataset_train_1 = datasets.MNIST(root='./data', download=True,train=False)
x_test, t_test = dataset_train_1.data.numpy(), dataset_train_1.targets.numpy()
x_test = x_train.reshape(10000,28*28)

train_size = x_train.shape[0]
iters_num = 10000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size,1)


network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch,t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.param[key] -= learning_rate*grad[key]

    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss) 
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch,t_batch)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

'''
当前的numerical_gradient方法无法正确计算梯度，因为损失函数没有正确绑定参数更新。
通过为每个参数创建独立的、能够临时修改该参数的损失函数，可以修复这个问题。
但需要注意的是，数值梯度法计算成本很高，对于实际训练建议使用反向传播算法。
'''