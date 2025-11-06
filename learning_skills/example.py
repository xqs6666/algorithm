import numpy as np

np_in = 100
n_samples = 1000

# 输入 x 的方差
x = np.random.randn(n_samples,np_in) # 方差1
var_x = np.var(x)
print(var_x)

# 权重 w 的方差
var_w_list = [0.01, 0.1, 1]  # 不同权重方差
for var_w in var_w_list:
    w = np.random.randn(np_in,np_in) * np.sqrt(var_w) # 权重初始化
    z = np.dot(x,var_w) 
    var_z = np.var(z)
    print(f"权重方差 Var(W)={var_w}, 输出方差 Var(z)={var_z:.2f}, 公式 n*Var(W)*Var(x)={np_in*var_w*var_x:.2f}")