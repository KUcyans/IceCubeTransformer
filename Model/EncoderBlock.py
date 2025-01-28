import torch
import torch.nn as nn

from BuildingBlocks.InnocentAttention import InnocentAttention
from BuildingBlocks.ScaledDotProductAttention import ScaledDotProductAttention
from BuildingBlocks.LayerNormalisation import LayerNormalisation
from BuildingBlocks.FFN import FFN

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_f, dropout=0.1, layer_idx=0, nan_logger=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(dropout)
        
        self.nan_logger = nan_logger
        
        # self.attention = InnocentAttention(self.d_model, self.n_heads, dropout)
        self.attention = ScaledDotProductAttention(self.d_model, self.n_heads, dropout, self.nan_logger)
        self.norm_attention = LayerNormalisation(self.d_model, self.nan_logger)
        
        self.ffn = FFN(self.d_model, self.d_f, dropout, self.nan_logger)
        self.norm_ffn = LayerNormalisation(self.d_model, self.nan_logger)
        
        
    def forward(self, x, mask=None):
        self.nan_logger.info(f"==============Entering Encoder Block {self.layer_idx}==============")
        
        # Attention
        attn_output = self.attention(x, mask)
        x = self.norm_attention(x + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_output))
        
        self.nan_logger.info(f"Encoder Block {self.layer_idx} output hasn't nan: {not torch.isnan(x).any()}")
        return x