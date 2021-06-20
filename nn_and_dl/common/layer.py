#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        return

    def forward(self, x):
        self.x = x
        Y = np.dot(x, W) + b
        return Y

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self,y self.t)

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.x) / batch_size
        return dx