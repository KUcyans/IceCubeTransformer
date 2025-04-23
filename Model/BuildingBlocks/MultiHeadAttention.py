import torch
import torch.nn as nn
import torch.nn.functional as F
from Enum.AttentionType import AttentionType
from Enum.PositionalEncodingType import PositionalEncodingType

from .ScaledDotProductAttention import ScaledDotProductAttention
from .InnocentAttention import InnocentAttention
from .ALiBiAttention import ALiBiAttention
from .T5Attention import T5Attention
# from .XFormersAttention import XFormersAttention
from rotary_embedding_torch import RotaryEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int, 
                 n_heads: int, 
                 attention_type: AttentionType,
                 positional_encoding_type: PositionalEncodingType,
                 dropout: float = 0.01):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type
        
        if self.attention_type == AttentionType.SDP:
            attention_cls = ScaledDotProductAttention
        elif self.attention_type == AttentionType.INNOCENT:
            attention_cls = InnocentAttention
        elif self.attention_type == AttentionType.ALIBI:
            attention_cls = ALiBiAttention
        elif self.attention_type == AttentionType.T5:
            attention_cls = T5Attention
        # elif attention_type == "xformers":
        #     attention_cls = XFormersAttention
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        self.attention_head = attention_cls(head_dim=self.head_dim, 
                                            n_heads=self.n_heads,
                                            dropout=dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.positional_encoding_type = positional_encoding_type
        if self.positional_encoding_type == PositionalEncodingType.ROPE:
            self.rope = RotaryEmbedding(dim=self.head_dim, use_xpos=True)
        
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
        
        ## ✅ Apply rotary embeddings
        if self.positional_encoding_type == PositionalEncodingType.ROPE: 
            self.rope = self.rope.to(q.device)
            q, k = self.rope.rotate_queries_and_keys(q, k)
        
        # ✅ Now q, k, v are ready to go
        attention_output = self.attention_head(q, k, v, event_length)
        if torch.isnan(attention_output).any():
            print(f"🚨 NaN detected AFTER attention!")
            print(f"🔍 Attention Output min/max: {attention_output.min().item()} / {attention_output.max().item()}")
            raise ValueError("NaN detected in attention output!")
        
        # concatenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        # attention_output = self.out_proj(attention_output)

        attention_output = self.dropout(attention_output)
        return attention_output