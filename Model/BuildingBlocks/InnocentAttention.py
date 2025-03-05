import torch
import torch.nn as nn
import torch.nn.functional as F

class InnocentAttention(nn.Module):
    def __init__(self, 
                 head_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim  

        # Output projection remains
        self.out_proj = nn.Linear(head_dim, head_dim)

        self.scale = torch.sqrt(torch.tensor(head_dim).float())
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_length=None):
        batch_size, seq_len, head_dim = q.shape

        # Compute attention scores
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply masking if event_length is provided
        if event_length is not None:
            row_indices = torch.arange(seq_len).view(1, -1, 1).to(q.device)
            col_indices = torch.arange(seq_len).view(1, 1, -1).to(q.device)
            expanded_event_length = event_length.view(-1, 1, 1).to(q.device)
            mask = (row_indices < expanded_event_length) & (col_indices < expanded_event_length)
            attention_weights = attention_weights.masked_fill(~mask, float('-1e9'))

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return output
