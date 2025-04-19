import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class T5Attention(nn.Module):
    def __init__(self, 
                 head_dim: int, 
                 n_heads: int,
                 dropout: float = 0.01):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = 4096 # < 5160  # Maximum sequence length for T5
        self.num_buckets = 32
        self.max_distance = 256

        # Learnable relative position embeddings
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    def forward(self, q, k, v, event_length=None):
        """
        q: (B, H, S, D)
        k: (B, H, S, D)
        v: (B, H, S, D)
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        k_t = k.transpose(-2, -1)  # (B, H, D, S)
        logits = torch.matmul(q, k_t) / math.sqrt(head_dim)  # (B, H, S, S)

        # Add relative position bias
        rel_bias = self._compute_bias(seq_len)  # (1, H, S, S)
        logits = logits + rel_bias

        # Apply optional attention mask
        if event_length is not None:
            mask = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len) < event_length.view(batch_size, 1, 1, 1)
            logits = logits.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, S, D)
        return out

    def _compute_bias(self, seq_len: int) -> torch.Tensor:
        """Compute learned relative position bias (1, H, S, S)."""
        device = self.relative_attention_bias.weight.device  # Ensure same device
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (S, S)

        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)  # (S, S, H)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, H, S, S)


    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """T5-style bucketing logic."""
        ret = 0
        n = -relative_position
        is_small = n < self.num_buckets // 2
        val_if_large = self.num_buckets // 2 + (
            (torch.log(n.float() / (self.num_buckets // 2)) /
             math.log(self.max_distance / (self.num_buckets // 2)) *
             (self.num_buckets - self.num_buckets // 2))
            .clamp(max=self.num_buckets - 1 - self.num_buckets // 2).to(torch.long)
        )
        ret = torch.where(is_small, n, val_if_large)
        ret = ret.clamp(min=0, max=self.num_buckets - 1)
        return ret
