import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, 
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

        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_v).transpose(1, 2)
        
        if torch.isnan(Q).any():
            print(f"NaN detected in Q!")
        if torch.isnan(K).any():
            print(f"NaN detected in K!")
        if torch.isnan(V).any():
            print(f"NaN detected in V!")

        # Q : [32, 8, 2421, 16] — batch_size, n_heads, seq_length, head_dim_qk
        # K : [32, 8, 2421, 16] — batch_size, n_heads, seq_length, head_dim_qk
        # V : [32, 8, 2421, 16] — batch_size, n_heads, seq_length, head_dim_v

        assert Q.shape == K.shape, "Q and K must have the same shape"
        
        # if mask is not None:
        #     # [batch_size, 1, 1, seq_length]
        #     mask = mask[:, None, None, :]
        
        # attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn_logits = attn_logits.clamp(min=-10, max=10)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)
            attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_logits = attn_logits.masked_fill(mask == 0, float('-1e9'))
        else:
            attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = F.scaled_dot_product_attention(
            query=Q, key=K, value=V,
            attn_mask=mask,
            scale=self.scale,
            dropout_p=self.dropout.p
        )

        # attn_output = F.scaled_dot_product_attention(
        #     query=Q, key=K, value=V,
        #     attn_mask=mask,
        #     scale=self.scale,
        #     dropout_p=self.dropout
        # )# Q @ K^T

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)

        self.nan_logger.info(f"---------- attention(scaled dot-product) ---------- ")
        self.nan_logger.info(f"attn_output hasn't nan: {not torch.isnan(attn_output).any()}")
        return self.out_proj(attn_output)

