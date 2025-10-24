import numpy as np
'''
如果正向传播时的输入值小于等于0，则反向传播的值为0。
因此，反向传播中会使用正向传播时保存的mask，将从上游传来的dout的
mask中的元素为True的地方设为0。
'''
class Relu:

    def __init__(self):
        self.mask = None
        pass

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    """
    反向传播
    dout: 上游传来的梯度
    """    
    def backward(self,dout):
        # dout 必须与 forward 的输出形状相同
        # 也就是与输入x的形状相同
        dout[self.mask] = 0
        dx = dout
        return dx
    
if __name__=="__main__":
    relu = Relu()
    x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
    dout = np.ones_like(x)

    out = relu.forward(x=x)
    dx = relu.backward(dout)

    print(out)
    print(dx)