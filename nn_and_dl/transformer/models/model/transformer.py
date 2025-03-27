import torch.nn as nn
import torch

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
            src_pad_idx, # input side pad index
            tgt_pad_idx, 
            tgt_sos_idx, # output side stop of sentence index
            src_vocab_size, 
            tgt_vocab_size, 
            d_model, 
            n_head, 
            max_len, 
            ffn_hidden, # feed forward network hidder layer dimensions
            n_layers,   # encoder and decoder layer count
            drop_prob,
            device):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.device = device

        self.encoder = Encoder(enc_vocab_size = src_vocab_size, 
                               max_len = max_len, 
                               d_model = d_model, 
                               ffn_hidden = ffn_hidden, 
                               n_head = n_head, 
                               n_layers = n_layers, 
                               drop_prob = drop_prob, 
                               device = device)

        self.decoder = Decoder(dec_vocab_size = tgt_vocab_size, 
                               max_len = max_len, 
                               d_model = d_model, 
                               ffn_hidden = ffn_hidden, 
                               n_head = n_head, 
                               n_layers = n_layers, 
                               drop_prob = drop_prob, 
                               device = device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)

        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

if __name__ == "__main__":
    src = torch.tensor([[13,4,24,0,123,3,3,3,3]])
    tgt = torch.tensor([[1,23,3,3,3,3,3,3,3,3]])

    transformer = Transformer(3, 3, 2, 1000, 2000, 128, 4, 10, 256, 6, 0, "cpu")
    # stop = False
    for i in range(1,len(tgt[0])):
        y = transformer(src, tgt)
        idx_i = torch.argmax(y[0][i], dim=-1)
        tgt[0][i] = idx_i
        if idx_i == 2:
            break

    print(tgt)
        
        


