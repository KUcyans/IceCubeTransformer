import torch
import torch.nn as nn
import xformers.ops as xops
from .AttentionHeadBase import AttentionHeadBase

class XFormersAttention(AttentionHeadBase):
    def __init__(self, 
                 head_dim: int, 
                 n_heads: int,
                 dropout: float = 0.01):
        super().__init__(head_dim=head_dim, 
                         n_heads=n_heads, 
                         dropout=dropout)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = dropout
    
    def forward(self, q, k, v, event_length=None):
        """
        from the MultiHeadAttention class
        q: batch_size, num_heads, seq_len, head_dim
        k: batch_size, num_heads, seq_len, head_dim
        v: batch_size, num_heads, seq_len, head_dim
        event_length: batch_size
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        assert num_heads == self.n_heads and head_dim == self.head_dim

        # permute the input tensors to (batch_size, seq_len, num_heads, head_dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # Optionally create an attention mask based on event_length
        attn_bias = None
        if event_length is not None:
            mask = torch.arange(seq_len, device=q.device).unsqueeze(0) < event_length.unsqueeze(1)

            attn_mask_bool = mask.unsqueeze(2) & mask.unsqueeze(1) #(batch_size, seq_len, seq_len)

            attn_mask_bool = attn_mask_bool.unsqueeze(1).expand(-1, num_heads, -1, -1) #(batch_size, num_heads, seq_len, seq_len)

            attn_bias = torch.zeros_like(attn_mask_bool, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(~attn_mask_bool, -torch.inf)

        output = xops.memory_efficient_attention(
            query=q, key=k, value=v,
            attn_bias=attn_bias,
            p=self.dropout
        )
        output = output.permute(0, 2, 1, 3).contiguous() 
        return output