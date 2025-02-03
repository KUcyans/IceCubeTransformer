import torch
import torch.nn as nn
import torch.nn.functional as F

class ALiBiAttention(nn.Module):
    def __init__(self, 
                 d_model: int, # the size of an embedded vector
                #  d_qk: int,
                #  d_v: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 nan_logger=None):
                     
        super().__init__()
        self.d_model = d_model # d_model
        self.d_qk = self.d_model
        self.d_v = self.d_model
        self.n_heads = n_heads
        self.head_dim_qk = self.d_qk // self.n_heads
        self.head_dim_v = self.d_v // self.n_heads
        self.scale = self.head_dim_qk ** -0.5
        self.dropout = nn.Dropout(dropout)

        # a layer is a linear transformation
        self.q_proj = nn.Linear(self.d_model, self.d_qk) # Projects input vectors of size d_model into query vectors of size d_qk using a weight matrix W_q of size (d_model x d_qk)
        self.k_proj = nn.Linear(self.d_model, self.d_qk)
        self.v_proj = nn.Linear(self.d_model, self.d_v)
        self.out_proj = nn.Linear(self.d_v, self.d_model)
        self.nan_logger = nan_logger

    # forward is invoked when calling the model
    # x is the input tensor
    # batch_size is the number of data samples in the batch
    # seq_length is the number of elements in the sequence(N_dom_max)
    # embed_dim is the dimension of the embedding
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        # print(f"batch_size: {batch_size}, seq_length: {seq_length}, embed_dim: {embed_dim}")
        # print(f"embed_dim: {embed_dim}, d_model: {self.d_model}")
        # assert embed_dim == self.d_model
        
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_v)
        attention_scores = self._get_attention_pure_score(x, batch_size, seq_length)
        alibi_bias = self._get_ALiBi_bias(x, seq_length)
        
        attention_scores += alibi_bias

        # Mask attention scores
        # masked_fill() fills elements of the tensor with a value where mask is True
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attention = F.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention, V).reshape(batch_size, seq_length, embed_dim)
        
        self.nan_logger.info(f"---------- attention(ALiBi) ---------- ")
        self.nan_logger.info(f"attn_output hasn't nan: {not torch.isnan(attention_output).any()}")
        
        return self.out_proj(attention_output)
    
    def _get_attention_pure_score(self, x, batch_size, seq_length):
        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        attention_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) * self.scale
        return attention_scores
    
    # HACK consider movig this outside this class. That would only be possible N_dom_max is constant for all parts...?
    def _get_ALiBi_bias(self, x, seq_length):
        # arange(n) returns a 1-D tensor of size n with values from 0 to n - 1
        slopes = 1.0 / (2 ** (torch.arange(self.n_heads).float() / self.n_heads))
        # to() moves the tensor to the device
        slopes = slopes.to(x.device)

        # view() reshapes the tensor
        relative_positions = torch.arange(seq_length).view(1, 1, seq_length) - torch.arange(seq_length).view(1, seq_length, 1)
        relative_positions = relative_positions.to(x.device)
        
        alibi_bias = slopes.view(self.n_heads, 1, 1) * relative_positions
        # unsqueeze() adds a dimension to the tensor
        alibi_bias = alibi_bias.unsqueeze(0)
        return alibi_bias