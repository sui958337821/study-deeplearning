import numpy as np

class MatMul(object):
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(self.params)]
        self.x = None

    def forward(self, x):
        self.x = x
        W, = self.params
        out = np.dot(x, W)

        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class SoftmaxWithLoss():
    def __init__(self):
        self.t = None
        self.loss = None
        self.y = None
        return

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0] # one-hot
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx