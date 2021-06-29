#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简单层的反向传播实现

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

# 加法层
class AddLayer:
    def __init__(self):
        return

    def forward(self, x, y):
        return x + y

    def backword(self, dout):
        return dout, dout