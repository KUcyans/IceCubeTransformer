import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_length=None):
        """
        from the MultiHeadAttention class
        q: batch_size, num_heads, seq_len, head_dim
        k: batch_size, num_heads, seq_len, head_dim
        v: batch_size, num_heads, seq_len, head_dim
        event_length: batch_size
        """
        batch_size, num_heads, seq_len, _ = q.shape
        attn_mask = None

        if event_length is not None:
            # Ensure mask shape matches the format expected by scaled_dot_product_attention
            row_indices = torch.arange(seq_len, device=q.device).view(1, 1, -1, 1)  # (1, 1, seq_len, 1)
            col_indices = torch.arange(seq_len, device=q.device).view(1, 1, 1, -1)  # (1, 1, 1, seq_len)

            expanded_event_length = event_length.view(batch_size, 1, 1, 1)  # (batch_size, 1, 1, 1)

            # Construct the attention mask (1 = keep, 0 = mask out)
            mask = (row_indices < expanded_event_length) & (col_indices < expanded_event_length)

            # Convert the mask to match PyTorch’s expected format for scaled_dot_product_attention
            attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=q.device)
            attn_mask.masked_fill_(~mask, float("-inf"))  # Invalidate unwanted positions

        output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)

        return output