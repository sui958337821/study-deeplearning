#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 4.0])
x + y
print(x+y)

# numpy操作
X = np.array([[51, 55], [14, 19], [0, 4]])
# 将X转换为一维数组
X = X.flatten()
print(X)
print(X[np.array([0,1,4])])

# 打印每个元素是否符合条件
print(X>15)

print(X[X>15])

# 绘制图形
import matplotlib.pyplot as plt
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label = "sin")
plt.plot(x, y2, label = "cos", linestyle = "--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.legend()
plt.show()

#显示图像
from matplotlib.image import imread

img = imread('./a.jpeg')
plt.imshow(img)
plt.plot(x, y1)
plt.show()
