import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size, head, seq_length, d_att = key.size()

        #k的转置
        key_t = key.transpose(2, 3)
        score = (query @ key_t) / math.sqrt(d_att)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)

        score = score @ value
        return score



