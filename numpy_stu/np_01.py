import numpy as np
import matplotlib.pyplot as plt

A = np.array([1,2,3,4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1,2], [3,4], [5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

A = np.array([[1,2], 
              [3,4]])
B = np.array([[5,6], 
              [7,8]])
print(np.dot(A,B))

print("--------------")

A = np.array([[1,2,3], 
              [4,5,6]])
B = np.array([[1,2], 
              [3,4], 
              [5,6]])
print(np.dot(A,B))

print("--------------")

A = np.array([[1,2], 
              [3,4], 
              [5,6]])
B = np.array([2,1])
print(np.dot(A,B))

print("--------------")

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
print(A1)
print(A1.shape)