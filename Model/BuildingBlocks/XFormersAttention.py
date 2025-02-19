import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops import fmha

class XFormersAttention(nn.Module):
    def __init__(self, d_model: int, 
                # d_qk: int,
                # d_v: int,
                 n_heads: int, 
                 dropout: float = 0.1, 
                 nan_logger=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.nan_logger = nan_logger

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_output = xops.memory_efficient_attention(Q, K, V)
        return self.out_proj(attn_output)