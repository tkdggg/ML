import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def softmax(z):
    exp = np.exp(z - np.max(z, axis=1).reshape(len(z), 1))
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    return exp / sum_exp


def one_hot(temp):
    one_list = np.zeros((len(temp), len(np.unique(temp))))
    one_list[np.arange(len(temp)), temp.astype(int).T] = 1
    return one_list


def compute_y_hat(W, X, b):
    return np.dot(X, W) + b


def cross_entropy(y, y_hat):
    return -(1 / len(y)) * np.sum(np.nan_to_num(y * np.log(y_hat + 1e-9)))


def predict(x):
    y_hat = softmax(compute_y_hat(W, x, bias))
    return np.argmax(y_hat, axis=1)[:, np.newaxis]


data = np.load('C:/Users/tkdg/Desktop/自然语言处理/自然语言处理实验二/mnist.npz')
(x_train, y_train), (x_test, y_test) = (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])
x_train = x_train.reshape(-1, 784)
X = x_train

np.random.seed(0)  # 定义随机数种子
W = np.random.rand(784, 10)
bias = np.random.rand(1, 10)

y = one_hot(y_train)
Num = 400  # 训练的次数
learn_rate = 0.001  # 学习率
loss_list = []

for i in range(1, Num):
    y_hat = compute_y_hat(W, X, bias)
    y_hat = softmax(y_hat)  # 将y_hat转换为概率
    loss = cross_entropy(y, y_hat)
    loss_list.append(loss)
    grad_w = (1 / len(X)) * np.dot(X.T, (y_hat - y))  # 计算梯度
    grad_b = (1 / len(X)) * np.sum(y_hat - y)
    W = W - learn_rate * grad_w  # 更新参数W
    bias = bias - learn_rate * grad_b  # 更新参数bias

plt.plot(loss_list)
x_test = x_test.reshape(-1, 784)
pre = predict(x_test)
pre = pre.T
print(pre)
print("测试正确率：", np.sum(pre == y_test) / len(y_test))
plt.show()

print("===============================================================================")

# 识别testimages文件夹下的手写图片
rec_arr = []
real_number = np.array([3, 9, 9, 8, 4, 1, 0, 6, 0, 9, 6, 8, 6, 1, 1, 9, 8, 9, 2, 3, 5, 5, 9, 4, 2, 1, 9, 4, 3, 9])
for j in range(30):
    image = Image.open(f'C:/Users/tkdg/Desktop/自然语言处理/自然语言处理实验二/testimages/{j}.jpg')  # 用PIL中的Image.open打开图像
    image_arr = np.array(image)  # 转化成numpy数组
    image_arr = image_arr.reshape(1, 784)
    image_arr = np.ravel(image_arr)  # 去掉多余的[]
    rec_arr.append(image_arr)

pre = predict(rec_arr)
pre = pre.T
print("识别出的30组数字为：")
print(pre)
print("识别正确率：", np.sum(pre == real_number) / len(real_number))
plt.plot(loss_list)