import numpy as np
import matplotlib.pyplot as plt

arr = np.array([2, 3, 0,-1])
print(np.maximum(0,arr))

arr1 = np.array([2, 3, 4,-1])
arr2 = np.array([1, 5, 2,-2])
print(np.maximum(arr1, arr2))

def ReLU(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.1)
y=ReLU(x)

plt.plot(x,y)
plt.savefig("ReLU.jpg")