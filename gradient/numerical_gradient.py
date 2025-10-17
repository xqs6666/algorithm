import numpy as np

# x^2+y^2=f(x,y)=z
def fun(x):
    return x[0]*x[0]+x[1]*x[1]

# 给一个点（x0,y0）求出偏导,如求x0偏导，y0看成常数不变
def numerical_gradient(fun,x):
    grad = np.zeros_like(x)
    h = 1e-4
    for i in range(len(x)):
        tmp = x[i]
        #f(x0+h) = (x0+h)^2+1
        x[i] = x[i]+h
        fun_x_add_h = fun(x)

        #f(x0-h) = (x0-h)^2+1
        x[i] = tmp
        x[i] = x[i]-h
        fun_x_reduce_h = fun(x)

        partial_derivative = (fun_x_add_h-fun_x_reduce_h)/(2*1e-4)
        grad[i] = partial_derivative
        x[i] = tmp
    return grad

print(numerical_gradient(fun,np.array([3.0, 4.0])))

def gradient_descent(fun,init_x,learing,step_num=100):
    x = init_x
    
    for idx in range(step_num):
        x = x-learing*numerical_gradient(fun=fun,x=x)

    return x

print(gradient_descent(fun,np.array([-3.0,4.0]),0.1))
print(gradient_descent(fun,np.array([-3.0,4.0]),10))
print(gradient_descent(fun,np.array([-3.0,4.0]),1e-4))