from torch import nn
from dataclasses import dataclass
import torch
import math
import torch.nn.functional as F
from typing import Optional, Tuple
class Llama3(nn.Module):
    def __init__(self, param):
        return
    
@dataclass
class ModelArgs:
    dim: int=4096
    n_layers: int=12
    n_heads: int=32
    # multi group query: kv heads
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    multiple_of: int = 256
    
    max_batch_size: int = 32
    max_seq_length: int = 2048
    
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
            
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    print(xq.shape)
    print(xk.shape)
    print(freqs_cis.shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return x
    bs, seq_len, n_kv_heads, head_dim = x.size()
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, num_repeats, head_dim)
        .reshape(bs, seq_len, n_kv_heads * num_repeats, head_dim)
    )
    
class Attention(nn.Module):
    def __init__(self, param: ModelArgs):
        super().__init__()
        
        self.n_local_kv_heads = param.n_kv_heads if param.n_kv_heads is not None else param.n_heads
        self.n_local_heads = param.n_heads
        self.n_kv_repeats = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = param.dim // self.n_local_heads
        
        self.w_q = nn.Linear(param.dim, param.dim)
        self.w_k = nn.Linear(param.dim, self.n_local_kv_heads * self.head_dim)
        self.w_v = nn.Linear(param.dim, self.n_local_kv_heads * self.head_dim)
        self.w_o = nn.Linear(param.dim, param.dim)
        
        self.cache_k = torch.zeros((param.max_batch_size, param.max_seq_length, self.n_local_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((param.max_batch_size, param.max_seq_length, self.n_local_kv_heads, self.head_dim))
        
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor],
    ):
        bsz, seq_len, _ = x.shape
        
        xq = self.w_q(x) # bsz, seq_len, dim
        xk = self.w_k(x)
        xv = self.w_v(x)
        
        print(xq.shape)
        print(xk.shape)
        print(self.cache_k.shape)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        

        # 将kv cache取出, 把新的续上
        self.cache_k[:bsz, start_pos : start_pos + seq_len] = xk
        self.cache_v[:bsz, start_pos : start_pos + seq_len] = xv
        
        #只取涉及本次计算的kv
        keys = self.cache_k[:bsz, : start_pos + seq_len]
        values = self.cache_v[:bsz, : start_pos + seq_len]
        
        # 重复kv,对齐heads
        keys = repeat_kv(keys, self.n_kv_repeats)
        values = repeat_kv(values, self.n_kv_repeats)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
                
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        return self.w_o(output)        
    
class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],

    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # ？？？
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        # ???
        self.linear3 = nn.Linear(dim, hidden_dim)
        
    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)) * self.linear3(x))

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        layer_id: int,
        param: ModelArgs,
    ):
        super().__init__()
        
        self.layer_id = layer_id,
        self.attention = Attention(param)
        self.ffn = FeedForward(
            param.dim, 
            param.dim * 4,
            multiple_of=param.multiple_of,
            ffn_dim_multiplier=param.ffn_dim_multiplier,)
        self.attention_norm = RMSNorm(param.dim, eps=param.norm_eps)
        self.ffn_norm = RMSNorm(param.dim, eps=param.norm_eps, )
        
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
                
        output = self.ffn(self.ffn_norm(h)) + h
        
        return output
            
class Transformer(nn.Module):
    def __init__(
        self,
        param: ModelArgs
    ):
        super().__init__()
        self.param = param
        self.n_layers = param.n_layers
        self.dim = param.dim
        self.vocab_size = param.vocab_size
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.output = nn.Linear(self.dim, self.vocab_size)
                
        self.layers = nn.ModuleList(
            [TransformerBlock(i, param) for i in range(self.n_layers)]
        )
        
        self.norm = RMSNorm(param.dim, eps = param.norm_eps)
        
        self.freqs_cis = precompute_freqs_cis(
            param.dim // param.n_heads,
            param.max_seq_length * 2,
            param.rope_theta,
        )
        
    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int
    ):
        bsz, seq_len = x.shape
        h = self.token_embedding(x)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        
        
        # 创建mask
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
            
            # 对角矩阵
            mask = torch.triu(mask, diagonal = 1)
            
            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device=x.device), mask]
            ).type_as(x)
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h)
        return output
    
if __name__ == "__main__":
    model_arg = ModelArgs(dim=128, n_layers=1, n_heads=8, n_kv_heads=4, vocab_size=1000)
    model = Transformer(model_arg)
    input = torch.LongTensor([[1,23],[33,23]])
    print(model(input, start_pos=7))
        
        
        
        