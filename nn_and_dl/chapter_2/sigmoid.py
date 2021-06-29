#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def step_function(x):
    return np.array(x>0, dtype=np.int)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y, linestyle="--")
# plt.ylim(-0.1, 1.1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)

def relu(x):
    return np.maximum(0, x)

# y = relu(x)
# plt.plot(x, y)
# plt.show()


def identity_function(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

# x = np.arange(-1, 10, 0.1)
# y = np.exp(x)
# plt.plot(x, y)
# plt.ylim(-1, 10)
# plt.show()