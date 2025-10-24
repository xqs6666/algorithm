import numpy as np

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        pass

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