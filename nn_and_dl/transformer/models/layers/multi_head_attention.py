import torch.nn as nn
from models.layers.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_att =  d_model // n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=False):
        # 1.dot product with the weight matrics
        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        
        query, key, value = self.split(query), self.split(key), self.split(value)

        out = self.attention(query, key, value, mask = mask)
        out = self.concat(out)
        out = self.w_concat(out)

        return out
    
    def split(self, tensor):
        batch_size, seq_len, d_model = tensor.size()

        tensor = tensor.view(batch_size, seq_len, self.n_head, self.d_att).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, n_head, sql_len, d_att = tensor.size()
        
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, sql_len, self.d_model)
        return tensor

if __name__ == "__main__":
    import torch
    from models.embedding.transformer_embedding import TransformerEmbedding
    src = torch.tensor([[223,45,12,2,23], [223,45,12,2,23]])
    embedding = TransformerEmbedding(1000, 128, 100, 0, "cpu")
    y = embedding.forward(src)
    # print(y.shape)
    self_attention = MultiHeadAttention(128, 4)
    q, k, v = y, y, y
    out = self_attention.forward(q, k, v, None)
    print(out.shape) # [2, 5, 128]
    print(out)