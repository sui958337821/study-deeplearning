## why deep?
## 为什么要引入损失函数？
## 梯度指示的方向
## 学习率是什么
η 表示更新量，在神经网络的学习中，称为学习率(learning rate)。学习率决定在一次学习中，应该学习多少，以及在多大程度上更新参数。
## 超参数
像学习率这样的参数称为超参数。这是一种和神经网络的参数(权重 和偏置)性质不同的参数。相对于神经网络的权重参数是通过训练 数据和学习算法自动获得的，学习率这样的超参数则是人工设定的。 一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利 进行的设定。
## 为什么不直接用非线性函数
## hiden size 和layer 如何确定
## epoch
epoch 是一个单位。一个 epoch 表示学习中所有训练数据均被使用过 一次时的更新次数。比如，对于 10000 笔训练数据，用大小为 100 笔数据的 mini-batch 进行学习时，重复随机梯度下降法 100 次，所 有的训练数据就都被“看过”了 A。此时，100 次就是一个 epoch。
计算图的特征是可以通过传递“局部计算”获得最终结果。“局部”这个 词的意思是“与自己相关的某个小范围”。局部计算是指，无论全局发生了什么， 都能只根据与自己相关的信息输出接下来的结果。
## 链式法则
如果某个函数由复合函数表示，则该复合函数的导数可以用构成复 合函数的各个函数的导数的乘积表示。
## 为什么反向传播算法能够更快地计算梯度？
反向传播的有点在于它仅利用一次前向传播就可以同时计算出所有的偏导数
## 为什么引入softmax
使用交叉熵误差作为 softmax 函数的损失函数后，反向传播得到 (y1−t1,y2−t2,y3−t3)这样“ 漂亮”的结果。实际上，这样“漂亮” 的结果并不是偶然的，而是为了得到这样的结果，特意设计了交叉熵误差函数。回归问题中输出层使用“恒等函数”，损失函数使用 “ 平 方 和 误 差 ”， 也 是 出 于 同 样 的 理 由 ( 3 . 5 节 )。 也 就 是 说 ， 使 用 “ 平
方和误差”作为“恒等函数”的损失函数，反向传播才能得到(y1 − t1, y2 − t2, y3 − t3)这样“漂亮”的结果。