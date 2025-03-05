import torch
import torch.nn as nn
import torch.nn.functional as F
from .ScaledDotProductAttention import ScaledDotProductAttention
from .InnocentAttention import InnocentAttention
from .ALiBiAttention import ALiBiAttention
# from .XFormersAttention import XFormersAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int, 
                 n_heads: int, 
                 attention_type: str = "scaled_dot", 
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        if attention_type == "scaled_dot":
            attention_cls = ScaledDotProductAttention
        elif attention_type == "innocent":
            attention_cls = InnocentAttention
        elif attention_type == "alibi":
            attention_cls = ALiBiAttention
        # elif attention_type == "xformers":
        #     attention_cls = XFormersAttention
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.multi_head_attention = nn.ModuleList(
            [attention_cls(head_dim=self.head_dim, 
                           dropout=dropout) for _ in range(n_heads)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, event_length=None):
        batch_size, seq_len, _ = x.shape

        # Project input to queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape into multiple heads and reorder dimensions
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply multi-head attention independently for each head
        head_outputs = [head(q[:, i], k[:, i], v[:, i], event_length) 
                        for i, head in enumerate(self.multi_head_attention)]
        
        # Concatenate the output of all heads
        multi_head_output = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, d_model)

        output = self.dropout(multi_head_output)

        return output
