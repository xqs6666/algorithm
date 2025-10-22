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
            c = np.max(x,axis=1,keepdims=True)
            exp_a_n = np.exp(x-c)
            exp_a_sum = np.sum(exp_a_n,axis=1,keepdims=True)
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
    
        def create_loss_fun(param_key):
            def loss_fun(param_value):
                original = self.param[param_key].copy()
                self.param[param_key] = param_value
                loss_val = self.loss(x,t)
                self.param[param_key] = original
                return loss_val
            return loss_fun

        grads = {}

        for key in ['W1','b1','W2','b2']:
            loss_fun = create_loss_fun(key)
            grads[key] = self.__numerical_gradient(loss_fun, self.param[key])
        return grads

dataset_train = datasets.MNIST(root='./data', download=True,train=True)
x_train, t_train = dataset_train.data.numpy(), dataset_train.targets.numpy()
x_train = x_train.reshape(60000,28*28)

dataset_train = datasets.MNIST(root='./data', download=True,train=False)
x_test, t_test = dataset_train.data.numpy(), dataset_train.targets.numpy()
x_test = x_test.reshape(10000,28*28)


iters_num = 10
train_size = x_train.shape[0]
batch_size = 1
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)


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
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))