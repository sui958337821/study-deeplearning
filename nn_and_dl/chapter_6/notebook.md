# 参数的更新
## SGD 随机梯度下降
使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠 近最优参数，这个过程称为随机梯度下降法(stochastic gradient descent)， 简称 SGD
### 缺点
虽然 SGD 简单，并且容易实现，但是在解决某些问题时可能没有效率
## Momentum
和 SGD 相比，我们发现 “之”字形的“程度”减轻了。这是因为虽然 x 轴方向上受到的力非常小，但 是一直在同一方向上受力，所以朝同一个方向会有一定的加速。反过来，虽 然 y 轴方向上受到的力很大，但是因为交互地受到正方向和反方向的力，它
们会互相抵消，所以 y 轴方向上的速度不稳定。因此，和 SGD 时的情形相比， 可以更快地朝 x 轴方向靠近，减弱“之”字形的变动程度。
## AdaGrad(Adaptive Grad)
### 学习率衰减
为学习率衰减(learning rate decay)的方法，即随着学习的进行，使学习率逐渐减小。实际上，一开始“多” 学，然后逐渐“少”学的方法，在神经网络的学习中经常被使用。
### 
AdaGrad 会为参数的每个元素适当地调整学习率，与此同时进行学习 (AdaGrad 的 Ada 来自英文单词 Adaptive，即“适当的”的意思)。AdaGrad 会记录过去所有梯度的平方和。因此，学习越深入，更新 的幅度就越小
## Adam
融合了Momentum和AdaGrad的特点

# 理解步骤
## 1.为什么各层的激活值要有合适的广度？