import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head =d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask=False):
        # x has shape: bs, seq_length, d_embed
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        # bs, seq_length, d_embed -> bs, seq_length, d_embed * 3 -> bs, seq_length, d_embed
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # bs, seq_length, d_embed -> bs, seq_length, H, d_embed / H -> bs, H, seq_length, d_embed / H
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # bs, H, seq_length, seq_length
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diag) is made up of 1's
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill with - inf
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (bs, H, seq_length, seq_length) @ (bs, H, seq_len, d_embed / H) -> (bs, H, seq_len, d_embed / H)
        output = weight @ v

        # (bs, H, seq_len, d_embed / H) -> (bs, seq_len, H, d_embed / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # bs, seq_len, d_embed
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int,
                 in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x, the query, the 'latent' : bs, seq_len_Q, dim_Q
        # y, the key-values, the 'content': bs, seq_len_KV, dim_KV = bs, 77, 768
        input_shape = x.shape
        bs, seq_length, d_embed = input_shape

        interim_shape = (bs, -1, self.n_heads, self.d_head)

        # Multiply query bs Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output