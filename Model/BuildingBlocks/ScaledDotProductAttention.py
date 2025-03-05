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

    def forward(self, x, event_lengths=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        batch_size, seq_len, input_dim = x.shape # batch size, sequence length, input dimension
        
        if event_lengths is not None:
            # row_indices = torch.arange(seq_len, device=x.device).view(1, -1, 1)
            # col_indices = torch.arange(seq_len, device=x.device).view(1, 1, -1)
            row_indices = torch.arange(seq_len).view(1, -1, 1).to(x.device) # Shape: (1, seq_len, 1)
            col_indices = torch.arange(seq_len).view(1, 1, -1).to(x.device) # Shape: (1, 1, seq_len)
            event_length_new = event_lengths.view(-1, 1, 1).to(x.device) # Shape: (batch_size, 1, 1)
            
            mask = (row_indices < event_length_new) & (col_indices < event_length_new)  
            # Shape: (batch_size, seq_len, seq_len)
            mask = ~mask  # Invert: False means attend, True means ignore
            mask = mask.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, seq_len)
            
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
        # this already has softmax and dropout applied
        
        return output

        