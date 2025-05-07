import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .AttentionHeadBase import AttentionHeadBase

class ALiBiAttention(AttentionHeadBase):
    def __init__(self, 
                 head_dim: int, 
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__(head_dim=head_dim, 
                         n_heads=n_heads, 
                         dropout=dropout)
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = 4096 # < 5160  # Maximum sequence length for ALiBi

        self._register_alibi_buffers(self.n_heads, self.max_seq_len)

    def forward(self, q, k, v, event_length=None):
        """
        q: (B, H, S, D)
        k: (B, H, S, D)
        v: (B, H, S, D)
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        k_t = k.transpose(-2, -1)  # (B, H, D, S)
        logits = torch.matmul(q, k_t) / math.sqrt(head_dim)  # (B, H, S, S)

        # Add ALiBi bias: slopes * rel_dist
        # bias = self.slopes * self.rel_dist[:seq_len, :seq_len]  # (H, S, S)
        bias = self._get_alibi_bias(seq_len)  # (1, H, S, S)
        logits = logits + bias  # (B, H, S, S)

        # Apply optional attention mask
        if event_length is not None:
            mask = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len) < event_length.view(batch_size, 1, 1, 1)
            logits = logits.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, S, D)
        return out

    def _register_alibi_buffers(self, n_heads, max_seq_len):
        slopes = self._get_alibi_slopes(n_heads).view(n_heads, 1, 1)
        self.register_buffer("slopes", slopes)  # (H, 1, 1)

        i = torch.arange(max_seq_len).view(1, -1)
        j = torch.arange(max_seq_len).view(-1, 1)
        rel_dist = (i - j).clamp(min=0)
        self.register_buffer("rel_dist", rel_dist)  # (S, S)

    
    def _get_alibi_slopes(self, n_heads):
        # Sourced from ALiBi paper code: https://github.com/ofirpress/attention_with_linear_biases/
        def get_slopes(n):
            def pow2(x): return 2 ** (-(2 ** -(math.log2(x) - 3)))
            if math.log2(n).is_integer():
                return torch.tensor([pow2(i + 1) for i in range(n)])
            else:
                closest_power = 2 ** math.floor(math.log2(n))
                base = torch.tensor([pow2(i + 1) for i in range(closest_power)])
                extra = self._get_alibi_slopes(2 * closest_power)[0::2][:n - closest_power]
                return torch.cat([base, extra], dim=0)
        return get_slopes(n_heads)

    def _get_alibi_bias(self, seq_len: int) -> torch.Tensor:
        """Compute the ALiBi bias term for attention logits."""
        bias = self.slopes * self.rel_dist[:seq_len, :seq_len]  # (H, S, S)
        return bias.unsqueeze(0)  # (1, H, S, S)
