import torch
import torch.nn as nn

from Model.BuildingBlocks.MultiHeadAttention import MultiHeadAttention
from Model.BuildingBlocks.LayerNormalisation import LayerNormalisation
from Model.BuildingBlocks.FFN import FFN
from Enum.AttentionType import AttentionType
from Enum.PositionalEncodingType import PositionalEncodingType

class EncoderBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 d_f: int,
                 attention_type: AttentionType,
                 positional_encoding_type: PositionalEncodingType,
                 dropout: float = 0.01,
                 layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding_type
        
        self.attention = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            attention_type=self.attention_type,
            positional_encoding_type=self.positional_encoding_type,
            dropout=dropout,
        )
        
        self.norm_attention = nn.LayerNorm(self.d_model)
        
        self.ffn = FFN(d_model=self.d_model, d_f=self.d_f, dropout=dropout)
        
        self.norm_ffn = nn.LayerNorm(self.d_model)
        
    def forward(self, x, event_length = None):
        # x shape: (batch_size, seq_len, d_model)
        attn_output = self.attention(x, event_length = event_length)
        if torch.isnan(x).any():
            print(f"🚨 NaN detected AFTER ATTENTION in layer {self.layer_idx}!")
            print(f"🔍 Min/Max: {x.min().item()} / {x.max().item()}")
            raise ValueError("NaN detected after attention!")
        
        x = x + attn_output
        # x shape: (batch_size, seq_len, d_model)
        x = self.norm_attention(x)
        
        ffn_output = self.ffn(x)
        if torch.isnan(x).any():
            print(f"🚨 NaN detected AFTER FFN in layer {self.layer_idx}!")
            print(f"🔍 Min/Max: {x.min().item()} / {x.max().item()}")
            raise ValueError("NaN detected after FFN!")
        
        x = x + ffn_output
        x = self.norm_ffn(x)
        return x
