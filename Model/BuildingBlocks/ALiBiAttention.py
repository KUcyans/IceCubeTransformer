import torch
import torch.nn as nn
import torch.nn.functional as F

class ALiBiAttention(nn.Module):
    def __init__(self, d_model: int, 
                # d_qk: int,
                # d_v: int,
                 n_heads: int,
                 dropout: float = 0.1,):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim)

        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) * self.scale
        attn_scores += self._get_ALiBi_bias(seq_length)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, V).reshape(batch_size, seq_length, self.d_model)

        return self.out_proj(attention_output)

    def _get_ALiBi_bias(self, seq_length):
        slopes = 1.0 / (2 ** (torch.arange(self.n_heads).float() / self.n_heads))
        relative_positions = torch.arange(seq_length).view(1, 1, seq_length) - torch.arange(seq_length).view(1, seq_length, 1)
        alibi_bias = slopes.view(self.n_heads, 1, 1) * relative_positions
        return alibi_bias.unsqueeze(0)
