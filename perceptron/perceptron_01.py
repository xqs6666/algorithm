import numpy as np

def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp<=theta:
        return 0
    if tmp>theta:
        return 1
print(AND(1,1))

def AND(x1,x2):
    w1,w2,theta=-0.5,-0.5,-0.7
    tmp=x1*w1+x2*w2
    if tmp<=theta:
        return 0
    if tmp>theta:
        return 1
print(AND(1,1))

x=np.array([0,1])
w=np.array([0.5,0.5])
b=-0.7
print(x*w)
result=np.sum(x*w)+b
print(result)

def AND(x1,x2):
    w1=0.5
    w2=0.5
    b=-0.7
    input=np.array([x1,x2])
    weight=np.array([w1,w2])
    if np.sum(input*weight)+b>0:
        return 1
    else:
        return 0

print(AND(1,0))

def NAND(x1,x2):
    w1=-0.5
    w2=-0.5
    b=0.7
    input=np.array([x1,x2])
    weight=np.array([w1,w2])
    if np.sum(input*weight)+b>0:
        return 1
    else:
        return 0
    
def OR(x1,x2):
    input=np.array([x1,x2])
    weight=np.array([0.5,0.5])
    b=-0.4
    tmp=np.sum(input*weight)+b
    if tmp<=0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(1,1))
print(XOR(1,0))
print(XOR(0,0))
print(XOR(0,1))