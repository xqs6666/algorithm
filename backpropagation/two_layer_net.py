from affine import Affine
from relu import Relu
from sigmoid import Sigmoid
from softmax_with_loss import SoftmaxWithLoss
import numpy as np
from collections import OrderedDict
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b1"] = np.zeros(output_size)

        self.layes = OrderedDict()
        self.layes["Affine1"] = Affine(self.params["W1"],self.params["b1"])
        self.layes["Relu1"] = Relu()
        self.layes["Affine2"] = Affine(self.params["W2"],self.params["b2"])
        
        self.lastLayer = SoftmaxWithLoss()
        pass

    def predict(self,x):
        data = x
        for layer in self.layes.values():
            data = layer.forward(data)
        return data

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    '''
    计算识别精度
    '''
    def accuray(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:
            t = np.argmax(t)
        accuray = np.sum(y == t)/float(t.shape[0])
        return accuray


    def gradient(self,x,t):
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layes.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layes["Affine1"].dW
        grads["b1"] = self.layes["Affine1"].db
        grads["W2"] = self.layes["Affine2"].dW
        grads["b2"] = self.layes["Affine2"].db
        
        return grads