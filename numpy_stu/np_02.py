import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
martix = np.arange(6).reshape(2,3)
print(martix)

it = np.nditer(martix,flags=["multi_index"],op_flags=["readwrite"])

while not it.finished:
    idx = it.multi_index
    print(idx)
    print(martix[idx])
    it.iternext()


dataset_train = datasets.MNIST(root='./data', download=True,train=True)
x_train, x_label = dataset_train.data.numpy(), dataset_train.targets.numpy()
x_train = x_train.reshape(60000,28*28)

print(x_train[[0]])
print(x_train[0])
print(x_train[[0]]==x_train[0])