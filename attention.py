import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 3 * dim) - > 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, H , dim/h) -> (batch_size, H, seq_len, dim/h)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # (batch_size, H, seq_len, dim/h) -> (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-2,-1)

        #applying mask: we substitute -infty to the interaction with the future tokens so that sofmax pushes to 0
        if causal_mask:
            # Mask where the upper traingle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight = weight.masked_fill(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight  = F.softmax(dim=-1)

        # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim/h) -> (batch_size, seq_len, H, dim/h)
        output = weight @ v

        # (batch_size, H, seq_len, dim/h) -> (batch_size, seq_len, h, dim/h)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (batch_size, seq_len, dim)
        return output
