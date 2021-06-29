#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
A = np.array([1,2,3])
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(np.ndim(B))
print(B.shape)

A = np.array([[1,2], [3,4]])
B = np.array([[1,1], [1,1]])
print(np.dot(A,B))