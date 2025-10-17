import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



# 1. 定义函数
def fun_1(x):
    return x[0]**2 + x[1]**2

# 2. 创建数据点网格
x0 = np.linspace(-5, 5, 100)  # x0在-5到5之间取100个点
x1 = np.linspace(-5, 5, 100)  # x1在-5到5之间取100个点
print(x0)
# 生成网格坐标矩阵
X0, X1 = np.meshgrid(x0, x1)

# 计算每个网格点上的函数值 Z = X0^2 + X1^2
Z = fun_1([X0, X1])

# 3. 创建图形和3D坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 4. 绘制曲面图
surf = ax.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.9, rstride=2, cstride=2)

# 5. 设置坐标轴标签
ax.set_xlabel('$x_0$', fontsize=12)
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_zlabel('$f(x_0, x_1)$', fontsize=12)
ax.set_title('3D Surface Plot of $f(x_0, x_1) = x_0^2 + x_1^2$', fontsize=14)

# 6. 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig("partial.jpg")



