import torch.nn as nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)
        
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(query=x, key=x, value=x, mask=src_mask)
        
        x = self.dropout1(x)

        x = self.norm1(x + _x)

        _x = x

        x = self.ffn(x)

        x = self.dropout2(x)

        x = self.norm2(x + _x)

        return x

if __name__ == "__main__":
    import torch
    from torch.nn.functional import pad
    src = torch.tensor([[13,4,24,0,123], [13,4,24,0,123]])
    from models.embedding.transformer_embedding import TransformerEmbedding
    transformer_emb = TransformerEmbedding(1000, 128, 100, 0, "cpu")
    y = transformer_emb.forward(src)
    encoder_layer = EncoderLayer(128, 4, 256, 0, "cpu")
    y = encoder_layer(y, None)
    print(y.size())
    print(y)
