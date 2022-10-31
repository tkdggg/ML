# 训练
import math
import numpy as np
import pandas as pd
import random
from pandas import DataFrame
# from pandas import Seres
import matplotlib.pyplot as plt


def sigmoid(x):  # 映射函数
    return 1 / (1 + np.exp(-x))


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

net_in = pd.read_csv("C:/Users/tkdg/Desktop/自然语言处理/实验一/BPdata_tr.txt")
net_in = np.array(net_in)
x = net_in[:, 0]
x_size = np.size(x)

y = net_in[:, 1]
y_size = np.size(y)
z = net_in[:, 2]
Z = np.zeros(len(z))
hidesize = 7  # 隐层神经元数量
W1x = np.random.random((hidesize, 1))  # 输入层与隐层之间的权重
W1y = np.random.random((hidesize, 1))  # 输入层与隐层之间的权重
B1 = np.random.random((hidesize, 1))  # 隐含层神经元的阈值
W2 = np.random.random((1, hidesize))  # 隐含层与输出层之间的权重
B2 = np.random.random((1, 1))  # 输出层神经元的阈值
threshold = 0.007  # 迭代速度
max_steps = 50  # 迭代最高次数
E = np.zeros((max_steps, 1))  # 误差随迭代次数的变化
Z = np.zeros((x_size, y_size))  # 模型的输出结果
los = []  # 误差
l = []  # 误差绝对值列表
for k in range(max_steps):
    temp = 0
    los = []
    for i in range(x_size):

        hide_in = np.dot(x[i], W1x) + np.dot(y[i], W1y) - B1  # 隐含层输入数据
        # print(x[i])
        hide_out = np.zeros((hidesize, 1))  # 隐含层的输出数据
        for m in range(hidesize):
            hide_out[m] = sigmoid(hide_in[m])  # 计算hide_out
            z_out = np.dot(W2, hide_out) - B2  # 模型输出
        Z[i] = z_out
        e = (z_out - z[i])
        los = np.append(los, abs(e))
        # 参数修改
        dB2 = -1 * threshold * e
        dW2 = e * threshold * np.transpose(hide_out)
        dB1 = np.zeros((hidesize, 1))
        for m in range(hidesize):
            dB1[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * (-1) * e * threshold)
        dW1x = np.zeros((hidesize, 1))
        dW1y = np.zeros((hidesize, 1))
        for m in range(hidesize):
            dW1y[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * y[i] * e * threshold)
        W1y = W1y - dW1y
        for m in range(hidesize):
            dW1x[m] = np.dot(np.dot(W2[0][m], sigmoid(hide_in[m])), (1 - sigmoid(hide_in[m])) * x[i] * e * threshold)
        W1x = W1x - dW1x
        B1 = B1 - dB1
        W2 = W2 - dW2
        B2 = B2 - dB2
        temp = temp + abs(e)
    l = np.append(l, sum(los) / x_size)
    E[k] = temp
    if k % int(max_steps / 10) == 0:
        print(k)
plt.title('误差变化')
plt.plot(l)


#验证
test_in=pd.read_csv("C:/Users/tkdg/Desktop/自然语言处理/实验一/BPdata_te.txt")
test_in=np.array(test_in)
x=test_in[:,0]
y=test_in[:,1]
z=test_in[:,2]
Z=np.zeros(len(z))
l=[]
for i in range(len(z)) :
    hide_in = np.dot(x[i], W1x) + np.dot(y[i], W1y) - B1  # 隐含层输入数据
    # print(x[i])
    hide_out = np.zeros((hidesize, 1))  # 隐含层的输出数据
    for m in range(hidesize):
        hide_out[m] = sigmoid(hide_in[m])  # 计算hide_out
        z_out = np.dot(W2, hide_out) - B2  # 模型输出
    Z[i] = z_out
    e = z_out - z[i]
    l=np.append(l,abs(e))
plt.plot(Z,'r')         #  由模型得到值
plt.plot(z)     #  真实值
plt.legend(["拟合值","真实值"])
print('平均误差',sum(l)/len(l))     #平均误差




#  使用sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
test_in=pd.read_csv("C:/Users/tkdg/Desktop/自然语言处理/实验一/BPdata_te.txt")
test_in=np.array(test_in)
net_in=pd.read_csv("C:/Users/tkdg/Desktop/自然语言处理/实验一/BPdata_tr.txt")
net_in=np.array(net_in)
X_train=net_in[:,:2]
y_train=net_in[:,2]
X_test=test_in[:,:2]
y_test=test_in[:,2]

scaler = StandardScaler() # 标准化转换
scaler.fit(X_train)  # 训练标准化对象
X_train = scaler.transform(X_train)   # 转换数据集

pf = MLPRegressor(solver='sgd', alpha=0.1,hidden_layer_sizes=(9, 7),max_iter=1000, random_state=0)
pf.fit(X_train, y_train)
pre = pf.predict(X_test)
meanlos=sum(abs(pre-y_test))/len(pre)
print('平均误差',meanlos)
plt.plot(pre,'r')
plt.plot(y_test)
plt.legend(["拟合值","真实值"])