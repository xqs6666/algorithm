import numpy as np
''' 
向前传播输入x 经过f(x) 输出y 保留y
反向传播时 用上一步传递的偏导 乘 当前(Sigmoid)函数的导数为 f'(x) = f(x) * (1 - f(x))，
其中 f(x)就是前向传播时计算出的输出值self.out
'''
class Sigmoid:
    def __init__(self):
        self.out = None
        pass

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        return dout*self.out*(1-self.out)