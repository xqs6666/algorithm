import  os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir)))
import numpy as np


class SimpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2,3)
        pass

    def cross_entropy_error_1(self,y,t):
        if y.ndim == 1:
            y = y.reshape(1,y.size)
            t = t.reshape(1,t.size)
        return -np.sum(t*np.log(y+1e-4))
    
    def cross_entropy_error_2(self,y,t):
        if y.ndim == 1:
            y = y.reshape(1,y.size)
            t = t.reshape(1,t.size)
        return -np.sum(np.log(y[np.arange(y.shape[0]),t]+1e-4))        
    
    def softmax(self,a):
        if a.ndim==2:
            c = np.max(a,axis=1,keepdims=True)
            exp_a = np.exp(a-c)
            exp_sum_a = np.sum(exp_a)
            return exp_a/exp_sum_a
        c = np.max(a)
        exp_a = np.exp(a-c)
        exp_sum_a = np.sum(exp_a)
        return exp_a/exp_sum_a
    
    def predict(self,x,w):
        return np.dot(x,w)
    
    def loss(self,x,t,w):
        z = self.predict(x=x,w=w)
        y = self.softmax(z)
        loss = self.cross_entropy_error_2(y,t)
        return loss
    
    def numerical_gradient(self,x,t):
        w = self.W
        zeros_w = np.zeros_like(w)

        for i in range(zeros_w.shape[0]):
            for j in range(zeros_w.shape[1]):
                tmp = w[i,j]
                h = 1e-4
                w[i,j] = w[i,j]+h
                fun_add_h = self.loss(x,t,w)

                w[i,j] = tmp-h
                fun_subtract_h = self.loss(x,t,w)

                partial = (fun_add_h-fun_subtract_h)/(2*h)
                zeros_w[i,j] = partial
                w[i,j] = tmp
        grad = zeros_w
        return grad
    

net = SimpleNet()
x = np.array([0.6,0.9])
t= np.array([2])
print(net.W)
print(net.loss(x,t,net.W))
print(net.numerical_gradient(x,t))
