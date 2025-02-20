import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, 
                # d_qk: int,
                # d_v: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = (self.d_model ** -0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits.clamp(min=-10, max=10)
        
        if mask is not None:
            mask = mask.to(Q.device)
            attn_logits = attn_logits.masked_fill(mask == 0, float('-1e9'))
        
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output
