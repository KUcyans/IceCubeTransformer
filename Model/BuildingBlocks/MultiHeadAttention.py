import torch
import torch.nn as nn
import torch.nn.functional as F
from .ScaledDotProductAttention import ScaledDotProductAttention
from .InnocentAttention import InnocentAttention
from .ALiBiAttention import ALiBiAttention
# from .XFormersAttention import XFormersAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int, 
                 n_heads: int, 
                 attention_type: str = "scaled_dot", 
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        if attention_type == "scaled_dot":
            attention_cls = ScaledDotProductAttention
        elif attention_type == "innocent":
            attention_cls = InnocentAttention
        elif attention_type == "alibi":
            attention_cls = ALiBiAttention
        # elif attention_type == "xformers":
        #     attention_cls = XFormersAttention
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        self.attention_head = attention_cls(head_dim=self.head_dim, dropout=dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, event_length=None):
        batch_size, seq_len, _ = x.shape

        # Project input into Q, K, V
        qkv = self.qkv_proj(x).view(batch_size, seq_len, self.n_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # q, k, v shape: (batch, seq, heads, head_dim)

        # ✅ Permute q, k, v before passing to attention
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        # k = k.permute(0, 2, 3, 1)  # (batch, heads, head_dim, seq)
        k = k.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        # ✅ Now q, k, v are ready to go
        attention_output = self.attention_head(q, k, v, event_length)
        if torch.isnan(attention_output).any():
            print(f"🚨 NaN detected AFTER attention!")
            print(f"🔍 Attention Output min/max: {attention_output.min().item()} / {attention_output.max().item()}")
            raise ValueError("NaN detected in attention output!")
        
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        attention_output = self.dropout(attention_output)
        return attention_output