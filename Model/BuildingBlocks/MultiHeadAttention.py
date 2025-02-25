import torch
import torch.nn as nn
import torch.nn.functional as F
from .ScaledDotProductAttention import ScaledDotProductAttention
from .InnocentAttention import InnocentAttention
from .ALiBiAttention import ALiBiAttention
# from .XFormersAttention import XFormersAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, 
                 n_heads: int, 
                 attention_type: str = "scaled_dot", 
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        if attention_type == "scaled_dot":
            attention_cls = ScaledDotProductAttention
        elif attention_type == "innocent":
            attention_cls = InnocentAttention
        elif attention_type == "alibi":
            attention_cls = ALiBiAttention
        elif attention_type == "xformers":
            attention_cls = XFormersAttention
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.multi_head_attention = nn.ModuleList(
            [attention_cls(d_model=self.d_model, dropout=dropout) for _ in range(n_heads)]
        )

        self.summarise = nn.Linear(d_model * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, event_length=None):
        head_outputs = [attention_head(x, event_length) for attention_head in self.multi_head_attention]
        multi_head_output = torch.cat(head_outputs, dim=-1)
        output = self.summarise(multi_head_output)
        output = self.dropout(output)
        return output