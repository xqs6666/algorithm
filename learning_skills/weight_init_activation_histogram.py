import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x = np.random.randn(1000,100)
node_num = 100 # 各隐藏层的节点（神经元）数
hidden_layer_size = 5 # 隐藏层有5层
activations = {} # 激活值的结果保存在这里

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    w = np.random.randn(node_num,node_num) * np.sqrt(2/node_num)
    z = np.dot(x, w)
    a = relu(z) # sigmoid函数
    activations[i] = a


for i, a in activations.items():
 plt.subplot(1, len(activations), i+1)
 plt.title(str(i+1) + "-layer")
 plt.hist(a.flatten(), 30, range=(0,1))
plt.savefig("标准差为1的高斯分布.png")