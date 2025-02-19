import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, 
                # d_qk: int,
                # d_v: int,
                 dropout: float = 0.1, 
                 nan_logger=None):
        super().__init__()
        self.d_model = d_model
        self.scale = (self.d_model ** -0.5)
        self.dropout = nn.Dropout(dropout)
        self.nan_logger = nan_logger

    def forward(self, Q, K, V, mask=None):
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits.clamp(min=-10, max=10)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-1e9'))
        
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        if self.nan_logger:
            self.nan_logger.info(f"---------- attention(Scaled Dot-Product) ---------- ")
            self.nan_logger.info(f"attn_output hasn't NaN: {not torch.isnan(attn_output).any()}")
        
        return attn_output
