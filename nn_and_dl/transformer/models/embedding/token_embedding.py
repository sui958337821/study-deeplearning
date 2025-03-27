import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

if __name__ == "__main__":
    import torch
    from torch.nn.functional import pad
    src = torch.tensor([[223,45,12,2,23], [223,45,12,2,23]])
    embedding = TokenEmbedding(128, 10000)
    y = embedding.forward(src)
    print(y.shape)
    print(y)
