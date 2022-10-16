import sys
sys.append("..")
from common.layers import MatMul, SoftmaxWithLoss
import numpy as np

class SimpleCbow(object):
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 生成层
        

        return
