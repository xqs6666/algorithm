import numpy as np

class AdaGradient:
    def __init__(self,learning=0.1):
        self.learning=learning
        self.r = {}
        pass

    def update(self,params,grads):
        if not self.r:
            for key,val in params.items():
                self.r[key] = np.zeros_like(val)
        for key,val in params.items():
            self.r[key] += grads[key]*grads[key]
            params[key] -= self.learning/(np.sqrt(self.r[key])+1e-7) * grads[key]
