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
        self.head_dim = d_model // n_heads
        
        # self.d_qk = d_model  # Kept for potential extension
        # self.d_v = d_model   # Kept for potential extension
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        if attention_type == "scaled_dot":
            self.attention_heads = ScaledDotProductAttention(
                d_model=self.head_dim, 
                # d_qk=self.head_dim,  # Kept for potential extension
                # d_v=self.head_dim,    # Kept for potential extension
                dropout=dropout)
        elif attention_type == "innocent":
            self.attention_heads = InnocentAttention(
                d_model=self.head_dim, 
                # d_qk=self.head_dim,  # Kept for potential extension
                # d_v=self.head_dim,    # Kept for potential extension
                n_heads=n_heads, 
                dropout=dropout)
        elif attention_type == "alibi":
            self.attention_heads = ALiBiAttention(
                d_model=self.head_dim, 
                # d_qk=self.head_dim,  # Kept for potential extension
                # d_v=self.head_dim,    # Kept for potential extension
                n_heads=n_heads, 
                dropout=dropout)
        elif attention_type == "xformers":
            self.attention_heads = XFormersAttention(
                d_model=self.head_dim, 
                # d_qk=self.head_dim,  # Kept for potential extension
                # d_v=self.head_dim,    # Kept for potential extension
                n_heads=n_heads, 
                dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        if mask is not None:
            mask = mask.to(x.device)
            mask = mask.bool()
            mask = ~mask
            mask = mask.unsqueeze(1)
        attn_output = self.attention_heads(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
        
        return self.out_proj(attn_output)
