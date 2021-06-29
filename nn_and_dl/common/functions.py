#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

def sofmax(x):
    if x.ndim == 2:
        x = x.T 
        x = x - np.max(x, axis = 0)

        y = np.exp(x) / np.sum(np.exp(x))

        return y.T

    x = x -np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))

    return y
    
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # 对于one-hot表示的监督数据，取其真实标签的位置，axis=1是按照每行，取每一行中最大元素的下标
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    loss = -np.sum(np.log(y[np.arange(batch_size), t] + 10e-7)) / batch_size
    return loss
