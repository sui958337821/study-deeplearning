#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.layers import *
from collections import OrderedDict
class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x):
        y = self.predict(x)
        return self.lastLayer.forward(y)

    def accuracy(self, x, t):
        y = self.predict(y)
        y = np.argmax(y, aixs = 1)
        if t.ndim != 1 : t = np.argmax(t, aixs = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W :self.loss(x, t)

        grads = {}
        
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads