import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        pass

    def softmax(self,x):
        np_max = np.max(x)
        np_x = np.exp(x-np_max)
        return np_x/np.sum(np_x)
    
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
        dx = (self.y - self.t) / batch_size
        return dx