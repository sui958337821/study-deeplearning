import torch.nn as nn
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Decoder, self).__init__()

        self.emb = TransformerEmbedding(vocab_size = dec_vocab_size, 
                                        d_model = d_model, 
                                        max_len = max_len, 
                                        drop_prob = drop_prob, 
                                        device = device)

        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model, 
                                                  n_head = n_head, 
                                                  ffn_hidden = ffn_hidden, 
                                                  drop_prob = drop_prob, 
                                                  device = device)
                                            for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(self, x, encoder_output, trg_mask, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, encoder_output, trg_mask, src_mask)

        output = self.linear(x)

        return output

if __name__ == "__main__":
    import torch
    vocab = torch.tensor([[13,4,24,0,123], [13,4,24,0,123]])
    from models.model.encoder import Encoder
    encoder = Encoder(1203, 10, 128, 256, 4, 6, 0, "cpu")
    y = encoder(vocab, None)
    print(y.shape)
    print(y)

    decoder = Decoder(1234, 10, 128, 256, 4, 6, 0, "cpu")
    vocab = torch.tensor([[13,4,24,2,123,2], [13,4,24,2,123,2]])
    y = decoder(vocab, y, None, None)

    print(y.shape)
    print(y)