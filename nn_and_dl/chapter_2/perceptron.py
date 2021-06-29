#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    y = np.array([-2, -2])
    b = 3
    tmp = np.sum(x * y) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 1, 1, -0.5
    tmp = w1 * x1 + w2 * x2 + theta
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    return AND(s1, s2)

print(XOR(0, 0))