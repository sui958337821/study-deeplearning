import torch.nn as nn
from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder, self).__init__()

        self.emb = TransformerEmbedding(vocab_size = enc_vocab_size, 
                                        d_model = d_model,
                                        max_len = max_len, 
                                        drop_prob = drop_prob, 
                                        device = device)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, 
                                                  n_head = n_head, 
                                                  ffn_hidden = ffn_hidden, 
                                                  drop_prob = drop_prob, 
                                                  device = device)
                                            for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

if __name__ == "__main__":
    import torch
    from torch.nn.functional import pad
    vocab = torch.tensor([[13,4,24,0,123], [13,4,24,0,123]])
    encoder = Encoder(1203, 10, 128, 256, 4, 6, 0, "cpu")
    y = encoder(vocab, None)
    print(y.shape)
    print(y)
    
