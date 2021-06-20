#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t,size)
    batch_size = y.shape[0]
    return -np.sum(np.sum(t * np.log(y + delta))) / batch_size

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x - h)) / (2 * h)

# 梯度
def numerical_gradient(f, x):
    grad = np.zeros_like(x)
    h = 1e-4
    for idx in x.size:
        x_tmp = x[idx]

        x[idx] = x_tmp + h
        delta_y1 = f(x)

        x[idx] = x_tmp - h
        delta_y2 = f(x)

        gard[idx] = (delta_y1 - delta_y2) / (2 * h)
        x[idx] = x_tmp
    return gard

# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        gard = numerical_gradient(f, init_x)
        x -= lr * gard

    return x