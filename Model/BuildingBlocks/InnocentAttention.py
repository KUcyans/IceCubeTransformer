import torch
import torch.nn as nn
import torch.nn.functional as F

class InnocentAttention(nn.Module):
    def __init__(self, 
                 head_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim  

        # self.out_proj = nn.Linear(head_dim, head_dim)

        self.scale = torch.sqrt(torch.tensor(head_dim).float())
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_length=None):
        """
        from the MultiHeadAttention class
        q: batch_size, num_heads, seq_len, head_dim
        k: batch_size, num_heads, seq_len, head_dim
        v: batch_size, num_heads, seq_len, head_dim
        event_length: batch_size
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        k = k.transpose(-2, -1)  # Ensure k is (batch_size, num_heads, head_dim, seq_len)
        attention_weights = torch.einsum("b h s d, b h d q -> b h s q", q, k) / self.scale

        if event_length is not None:
            row_indices = torch.arange(seq_len).view(1, 1, -1, 1).to(q.device)  # (1, 1, seq_len, 1)
            col_indices = torch.arange(seq_len).view(1, 1, 1, -1).to(q.device)  # (1, 1, 1, seq_len)

            expanded_event_length = event_length.view(batch_size, 1, 1, 1).to(q.device)  # (batch_size, 1, 1, 1)

            mask = (row_indices < expanded_event_length) & (col_indices < expanded_event_length)
            attention_weights = attention_weights.masked_fill(~mask, -1e9)

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # ✅ Direct einsum without permute
        output = torch.einsum("b h s q, b h q d -> b h s d", attention_weights, v)

        return output