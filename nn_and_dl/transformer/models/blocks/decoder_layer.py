import torch.nn as nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob, device):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.en_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, encoder_output, trg_mask, src_mask):
        _x = x
        x = self.self_attention(query=x, key=x, value=x, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if encoder_output is not None:
            _x = x
            x = self.en_dec_attention(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return  x

if __name__ == "__main__":
    import torch
    from torch.nn.functional import pad
    src = torch.tensor([[13,4,24,0,123], [13,4,24,0,123]])
    from models.embedding.transformer_embedding import TransformerEmbedding
    transformer_emb = TransformerEmbedding(1000, 128, 100, 0, "cpu")
    y = transformer_emb.forward(src)
    from models.blocks.encoder_layer import EncoderLayer
    encoder_layer = EncoderLayer(128, 4, 256, 0, "cpu")
    encoder_output = encoder_layer(y, None)
    print(encoder_output)
    dst = torch.tensor([[0,3,3,3,3], [0,3,23,3,3]])
    dst = transformer_emb.forward(dst)
    decoder_layer = DecoderLayer(128, 4, 256, 0, "cpu")
    decoder_output = decoder_layer(dst, encoder_output, None, None)
    print(decoder_output)