import numpy as np
import matplotlib.pyplot as plt



def step_function(x):
    return np.array(x>0,dtype=int)

x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
plt.savefig('step_function.png')


def sigmoid_function(x):
    return 1 / (1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)
y=sigmoid_function(x=x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid_function.png')


