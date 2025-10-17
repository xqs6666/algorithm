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
        c = np.max(a)
        exp_a = np.exp(a-c)
        exp_sum_a = np.sum(exp_a)
        return exp_a/exp_sum_a
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x=x)
        y = self.softmax(z)
        loss = self.cross_entropy_error_2(y,t)
        return loss

net = SimpleNet()
# print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
print(np.argmax(p))
loss = net.loss(x,np.array([2]))
print(loss)
# exp_a = np.exp(net.predict(x)-np.max(net.predict(x)))
# exp_sum_a = np.sum(exp_a)
# print(exp_a/exp_sum_a)