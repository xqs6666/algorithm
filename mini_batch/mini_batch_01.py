from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

dataset_train = datasets.MNIST(root='./data', download=True, train=True)
x_train = dataset_train.data.numpy().astype(np.float32)
x_train = x_train.reshape(60000,784)
x_label = dataset_train.targets.numpy()
t_train = []
for idx in range(len(x_label)):
    var = [i*0 for i in range(10)]
    for i in range(0,10):
        if i==x_label[idx]:
            var[i] = 1
        else:
            var[i] = 0
    t_train.append(var)
t_train = np.array(t_train)
    

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)#0-train_size 选10个数据


# 真实标签（one-hot编码）
t_batch = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 真实类别：1
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 真实类别：4  
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   # 真实类别：5
])

# 模型预测概率（softmax输出）
y_batch = np.array([
    [0.1, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0],    # 对类别1预测概率0.6
    [0, 0.3, 0.2, 0, 0.5, 0, 0, 0, 0, 0],    # 对类别4预测概率0.5
    [0.6, 0.3, 0, 0, 0, 0.1, 0, 0, 0, 0]     # 对类别5预测概率0.1（预测错误）
])

print(t_batch*y_batch)
print(y_batch.shape[0])
print(np.sum(t_batch*y_batch)/y_batch.shape[0])
print(np.log(y_batch+1e-7))

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.reshape)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

print(cross_entropy_error(y_batch,t_batch))

# 实例数据：3个样本的mini-batch，5个类别
# 真实标签（直接使用类别编号，非one-hot编码）
t_labels = np.array([2, 0, 3])  # 样本1真实类别2，样本2真实类别0，样本3真实类别3

# 模型预测概率（softmax输出）
y_pred = np.array([
    [0.1, 0.2, 0.6, 0.05, 0.05],  # 样本1：对类别2预测概率0.6
    [0.7, 0.1, 0.1, 0.05, 0.05],  # 样本2：对类别0预测概率0.7
    [0.1, 0.1, 0.2, 0.5, 0.1]     # 样本3：对类别3预测概率0.5
])

idx = np.arange(y_pred.shape[0])
correct_labels = t_labels

correct_probs  = y_pred[idx,correct_labels]
print(correct_probs)

def cross_entyopy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.reshape)
    batch_size = y.shape[0]
    correct_probs = y[np.arange(y.shape[0]),t]#获取y_pred每一行对应t位置的正确概率是多少
    return -np.sum(np.log(correct_probs+1e-7)/batch_size)

print(cross_entyopy_error(y_pred,t_labels))


x = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
# 分别获取 (0,0), (1,1), (2,0) 位置的元素
data = x[[0,1,2],[0,1,0]]
print(data)

true_data = [0,0,1]
data = x[np.arange(x.shape[0]),true_data]
