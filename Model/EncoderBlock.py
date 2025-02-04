import torch
import torch.nn as nn

from Model.BuildingBlocks.InnocentAttention import InnocentAttention
from Model.BuildingBlocks.ALiBiAttention import ALiBiAttention
from Model.BuildingBlocks.ScaledDotProductAttention import ScaledDotProductAttention
from Model.BuildingBlocks.XFormersAttention import XFormersAttention
from Model.BuildingBlocks.LayerNormalisation import LayerNormalisation
from Model.BuildingBlocks.FFN import FFN

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_f, dropout=0.1, nan_logger=None, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_f = d_f
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(dropout)
        
        self.nan_logger = nan_logger
        
        # self.attention = InnocentAttention(
        #     d_model = self.d_model, 
        #     n_heads = self.n_heads, 
        #     dropout = dropout, 
        #     nan_logger = self.nan_logger)
        
        # self.attention = ALiBiAttention(
        #     d_model = self.d_model, 
        #     n_heads = self.n_heads, 
        #     dropout = dropout, 
        #     nan_logger = self.nan_logger)
        
        self.attention = ScaledDotProductAttention(
            d_model = self.d_model, 
            n_heads = self.n_heads, 
            dropout = dropout, 
            nan_logger = self.nan_logger)
        
        # self.attention = XFormersAttention(
        #     d_model = self.d_model, 
        #     n_heads = self.n_heads, 
        #     dropout = dropout, 
        #     nan_logger = self.nan_logger)
        
        self.norm_attention = LayerNormalisation(d_n = self.d_model, 
                                                 nan_logger = self.nan_logger)
        
        self.ffn = FFN(d_model = self.d_model, 
                       d_f = self.d_f, 
                       dropout = dropout, 
                       nan_logger = self.nan_logger)
        
        self.norm_ffn = LayerNormalisation(d_n = self.d_model, 
                                           nan_logger = self.nan_logger)
        
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