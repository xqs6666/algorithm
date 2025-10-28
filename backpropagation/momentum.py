import numpy as np

class Momentum:
    def __init__(self,momentum=0.9,learning=0.01):
        self.v = {}
        self.momentum = momentum
        self.learning= learning
        pass

    def update(self,parames,grad):
        if self.v == None:
            for key,val in parames.items():
                self.v[key] = np.zeros_like(val)
        
        for key,val in parames.items():
            self.v[key] = self.momentum * self.v[key] - self.learning*grad[key]
            parames[key] +=  self.v[key]

            