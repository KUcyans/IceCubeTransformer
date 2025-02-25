import torch
import torch.nn as nn

from Model.BuildingBlocks.MultiHeadAttention import MultiHeadAttention
from Model.BuildingBlocks.LayerNormalisation import LayerNormalisation
from Model.BuildingBlocks.FFN import FFN

class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 d_f: int,
                 attention_type: str, 
                 dropout: float = 0.1,
                layer_idx: int = 0,):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(dropout)
        
        self.attention = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            attention_type=attention_type,
            dropout=dropout,
        )
        
        self.norm_attention = nn.LayerNorm(self.d_model)
        
        self.ffn = FFN(d_model=self.d_model, d_f=self.d_f, dropout=dropout)
        
        self.norm_ffn = nn.LayerNorm(self.d_model)
        
    def forward(self, x, event_length = None):
        attn_output = self.attention(x, event_length = event_length)
        x = x + attn_output
        x = self.norm_attention(x)
        
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm_ffn(x)
        return x