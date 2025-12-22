# 用梯度下降算法求函数最小值：L=x1^2 +2x2^2 最优化
import matplotlib.pyplot as plt

lmd = 0.1 # 0.001, 0.1, 1, 1不行 会形成振荡不收敛 (\lambda) 尝试
x1 = -10  # 1
x2 = 30  # 3
L1 = x1 ** 2 + 2 * x2 ** 2 
n = 0
loss_delta = 1
threshold = 0.0000000001
loss_deltas = []

while loss_delta > threshold and n <= 1999: 
    # n=0开始循环,n <= 1999 这个条件其实是循环2000次
    x1 = x1 - 2 * lmd * x1  # 迭代
    x2 = x2 - 4 * lmd * x2  # 迭代
    L2 = x1 ** 2 + 2 * x2 ** 2
    n = n + 1
    print(f'{n}: ({x1}, {x2})')
    print("\n")

    loss_delta = abs(L2 - L1)
    # 计算前后两次迭代后函数差值的绝对值
    loss_deltas.append(loss_delta)
    L1 = L2       

print(x1, x2, L1, n)
print(f'\n循环了 {n} 次，得到如下结果:\n x1 = {x1}, x2 = {x2} 时,\nmin L = {L1}')

plt.plot(loss_deltas)
plt.show()

a = 1
