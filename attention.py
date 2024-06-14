import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads:int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        #the output of in_proj is then chunked into q,k,v (that's why we multiply by 3)
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads  = n_heads
        self.d_head   = d_embed // n_heads
    
    # note: mask is a way to relate a particular token only to those tokens that come before it
    def forward(self, x:torch.Tensor, causal_mask=False):
        #x: (batch_size, seq_len, dim)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj()

