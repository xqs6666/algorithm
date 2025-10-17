import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def fun(x):
    return x[0]**2 + x[1]**2

# 生成网格点
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# 当前点和梯度
point = np.array([3.0, 4.0])
grad = np.array([6.0, 8.0])

# 绘制等高线
plt.figure(figsize=(8, 7))
contours = plt.contour(X, Y, Z, levels=np.arange(0, 60, 5), cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8, fmt="z=%.0f")

# 绘制当前点
plt.scatter(point[0], point[1], color='red', s=80, label='点 (3,4)')

# 绘制梯度箭头（红色）
plt.quiver(point[0], point[1], grad[0], grad[1],
           angles='xy', scale_units='xy', scale=1, color='red', width=0.008)
plt.text(point[0]+0.3, point[1]+0.3, "梯度方向 (6,8)", color='red', fontsize=10)

# 绘制梯度反方向（蓝色，即下降方向）
plt.quiver(point[0], point[1], -grad[0], -grad[1],
           angles='xy', scale_units='xy', scale=1, color='blue', width=0.008)
plt.text(point[0]-5.2, point[1]-3, "下降方向 (-6,-8)", color='blue', fontsize=10)

# 坐标系与样式
plt.title("函数 $f(x,y)=x^2+y^2$ 的梯度几何意义", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axis('equal')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

plt.savefig("grad.jpg")
