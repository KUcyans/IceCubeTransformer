import torch
import torch.nn as nn
import torch.nn.functional as F

class InnocentAttention(nn.Module):
    def __init__(self, 
                 d_model: int, # the size of an embedded vector
                 # d_qk: int,
                    # d_v: int,
                    n_heads: int,
                    dropout: float = 0.1,
                    nan_logger=None):
        super().__init__()
        self.d_model = d_model
        self.d_qk = self.d_model
        self.d_v = self.d_model
        self.n_heads = n_heads
        self.head_dim_qk = self.d_qk // self.n_heads
        self.head_dim_v = self.d_v // self.n_heads
        self.scale = self.head_dim_qk ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(self.d_model, self.d_qk)
        self.k_proj = nn.Linear(self.d_model, self.d_qk)
        self.v_proj = nn.Linear(self.d_model, self.d_v)
        self.out_proj = nn.Linear(self.d_v, self.d_model)
        self.nan_logger = nan_logger
        
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_v)
        attn_scores = self._get_attention_pure_score(x, batch_size, seq_length)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        
        attention = F.softmax(attn_scores, dim=-1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention, V).reshape(batch_size, seq_length, embed_dim)
        
        self.nan_logger.info(f"---------- attention(Innocent) ---------- ")
        self.nan_logger.info(f"attn_output hasn't nan: {not torch.isnan(attention_output).any()}")
        return self.out_proj(attention_output)
    
    def _get_attention_pure_score(self, x, batch_size, seq_length):
        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        attention_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) * self.scale
        return attention_scores