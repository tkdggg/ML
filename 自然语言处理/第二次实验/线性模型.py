import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# 构造损失函数
def loss(y, y_hat):
    return np.sum((y - y_hat) ** 2)

# 读取数据
data = pd.read_csv('C:/Users/tkdg/Desktop/自然语言处理/自然语言处理实验二/line_fit_data.csv')
data=np.array(data)
x = data[:80, 0]   # 前80条数据用于训练
y = data[:80, 1]
w = random.random()
b = random.random()
learning_rate = 0.1
los = 0
loss_list = []

for i in range(1, 80):
    y_hat = w * x + b#构造一个线性模型
    los = loss(y, y_hat)
    loss_list.append(los)
    # 最小化方差（训练）
    grad_w = -2 * (y_hat - y) * x
    grad_b = -2 * (y_hat - y)
    w = w + learning_rate * grad_w
    b = b + learning_rate * grad_b
print('参数：w=%s，b=%s'%(w[-1],b[-1]))
plt.plot(loss_list)


# 性能评估
x = data[80:, 0]
y = data[80:, 1]
Y=[]
for i in range(0, 20):
    Y = np.append(Y,w[-1] * x[i] + b[-1])#构造一个线性模型
plt.plot(Y,'r')
plt.plot(y)
print('平均误差：',sum(abs(Y-y)/len(y)))