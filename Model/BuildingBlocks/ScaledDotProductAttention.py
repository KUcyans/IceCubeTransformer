import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, 
                # d_qk: int,
                # d_v: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = torch.sqrt(torch.tensor(self.d_model).float())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, event_lengths=None, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        batch_size, seq_len, d_model = x.shape
        attention_weights =  torch.matmul(q, k.transpose(-2, -1)) # compute attention weights by taking the dot product of query and key
        attention_weights = attention_weights / self.scale # scale the attention weights
        
        if event_lengths is not None:
            # Step 1: Generate row and column indices for a square matrix of size seq_len
            row_indices = torch.arange(seq_len).view(1, -1, 1)  # Shape: (1, seq_len, 1)
            col_indices = torch.arange(seq_len).view(1, 1, -1)  # Shape: (1, 1, seq_len)

            # Map row and col indices to the device of the input tensor
            row_indices = row_indices.to(x.device)
            col_indices = col_indices.to(x.device)

            # Step 2: Compare indices against event_lengths to create the mask
            event_lengths_new = event_lengths.view(-1, 1, 1).to(x.device)  # Shape: (batch_size, 1, 1)

            mask = (row_indices < event_lengths_new) & (col_indices < event_lengths_new)
            # Mask shape: (batch_size, seq_dim, seq_dim)
            #attention_weights[~mask] = float('-inf')
            attention_weights = attention_weights.masked_fill(~mask, float('-1e9'))
        
        attention_weights = F.softmax(attention_weights, dim=-1) # apply softmax to get the attention weights
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        
        return output