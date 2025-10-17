import numpy as np

a_n = np.array([0.1,0.3,0.4])
exp_a_n = np.exp(a_n)
exp_a_sum = np.sum(exp_a_n)


def softmax_1(a):
    exp_a_n = np.exp(a)
    exp_a_sum = np.sum(exp_a_n)
    return exp_a_n/exp_a_sum

print(softmax_1(a_n))


a = np.array([1010,1000,990])
max_a = np.max(a)
y = np.exp(a-max_a) / (np.sum(np.exp(a-max_a)))
print(y)

def softmax_2(a):
    c = np.max(a)
    exp_a_n = np.exp(a-c)
    exp_a_sum = np.sum(exp_a_n)
    return exp_a_n/exp_a_sum

print(np.sum(softmax_2(a))) #1

a = np.array([0.3, 2.9, 4.0])
y = softmax_2(a)
print(y)