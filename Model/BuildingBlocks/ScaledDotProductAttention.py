import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, 
                 head_dim: int, 
                 n_heads: int,
                 dropout: float = 0.01):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_length=None):
        """
        from the MultiHeadAttention class
        q: batch_size, num_heads, seq_len, head_dim
        k: batch_size, num_heads, seq_len, head_dim
        v: batch_size, num_heads, seq_len, head_dim
        event_length: batch_size
        """
        batch_size, _, seq_len, _ = q.shape
        attn_mask = None
        
        # clamp_value = 1e2
        # q = torch.clamp(q, -clamp_value, clamp_value)
        # k = torch.clamp(k, -clamp_value, clamp_value)
        # v = torch.clamp(v, -clamp_value, clamp_value)
        
        if event_length is not None:
            row_indices = torch.arange(seq_len, device=q.device).view(1, 1, -1, 1)  # (1, 1, seq_len, 1)
            col_indices = torch.arange(seq_len, device=q.device).view(1, 1, 1, -1)  # (1, 1, 1, seq_len)

            expanded_event_length = event_length.view(batch_size, 1, 1, 1)  # (batch_size, 1, 1, 1)

            mask = (row_indices < expanded_event_length) & (col_indices < expanded_event_length)

            attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=q.device)
            attn_mask.masked_fill_(~mask, float("-1e9"))  # Invalidate unwanted positions

        output = F.scaled_dot_product_attention(query=q, key=k, value=v,
                                                attn_mask=attn_mask,
                                                dropout_p=self.dropout.p)
        # scale factor is 1/qrt(head_dim) by default in scaled_dot_product_attention

        return output