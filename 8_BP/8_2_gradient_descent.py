# 用梯度下降算法求函数最小值：L=x1^2 +2x2^2
import matplotlib.pyplot as plt

m = 0.1 # 0.001, 0.1, 1, 1不行 会形成振荡不收敛
x1 = -10  # 1
x2 = 30  # 3
L = x1 ** 2 + 2 * x2 ** 2
n = 0
loss_delta = 1
threshold = 0.0000000001
loss_deltas = []

while loss_delta > threshold and n < 2000:
    x1 = x1 - 2 * m * x1  # 迭代
    x2 = x2 - 4 * m * x2  # 迭代
    loss_delta = abs(x1 ** 2 + 2 * x2 ** 2 - L)
    # 计算前后两次迭代后函数差值的绝对值
    loss_deltas.append(loss_delta)
    L = x1 ** 2 + 2 * x2 ** 2
    print(x1, x2, n)
    print("\n")
    n = n + 1

print(x1, x2, L, n)

plt.plot(loss_deltas)
plt.show()

a = 1
