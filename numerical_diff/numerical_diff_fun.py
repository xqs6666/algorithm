import numpy as np
import matplotlib.pylab as plt

def numerical_diff(fun,x_0):
    h = 1e-4
    return (fun(x_0+h)-fun(x_0-h))/(2*h)

def fun_x1(x):
    return x**2

def fun_x2(x):
    return 0.01*(x**2)+0.1*x

x=np.arange(0.0,20.0,0.1)
y=fun_x2(x=x)
plt.plot(x,y)
plt.savefig("0.01*x**2 + 0.1*x.jpg")
print(numerical_diff(fun_x1,1))
print(numerical_diff(fun_x2,1))

def partial_derivative_fun(x):
    return x[0]*x[0]+x[1]*x[1]

def partial_derivative_tmp1_fun(x0):
    return x0*x0+4.0**2.0

print(numerical_diff(partial_derivative_tmp1_fun,3.0))