import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops import fmha


class XFormersAttention(nn.Module):
    def __init__(self, d_model: int, 
                #  d_qk: int,
                #  d_v: int,
                 n_heads: int, 
                 dropout: float = 0.1, 
                 nan_logger=None,
                 verbosity=0):
        super().__init__()
        self.d_model = d_model
        self.d_qk = self.d_model
        self.d_v = self.d_model
        self.n_heads = n_heads
        self.head_dim_qk = self.d_qk // self.n_heads
        self.head_dim_v = self.d_v // self.n_heads
        self.scale = self.head_dim_qk ** -0.5  # Scaling factor for stability
        self.dropout = dropout
        self.nan_logger = nan_logger

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.verbosity = verbosity

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape

        Q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        K = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_qk)
        V = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim_v)
        # Q = Q.to(torch.bfloat16)
        # K = K.to(torch.bfloat16)
        # V = V.to(torch.bfloat16)
        
        assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary (0 or 1)."
        
        # shape of the mask: [batch_size, n_heads, seq_length, seq_length]
        if self.verbosity > 1:
            print(f" mask shape should be [batch_size, n_heads, seq_length, seq_length]")
            print(f" batch_size: {batch_size}, n_heads: {self.n_heads}, seq_length: {seq_length}, embed_dim: {embed_dim}")
            # batch_size: 3, 1, seq_length: 2421, embed_dim: 128
            print(f" the shape of the mask is {mask.shape}")
            # the shape of the mask is torch.Size([3, 2421])
        
        # if mask is not None:
        #     seqlens = mask.sum(dim=1).long().tolist()
        #     attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens)
        #     attn_bias = attn_bias.to(x.device)
        #     # print(f" attn_bias shape: {attn_bias.shape}")
        # else:
        #     attn_bias = None

        # attn_output = xops.memory_efficient_attention(
        #     query=Q,
        #     key=K,
        #     value=V,
        #     attn_bias=attn_bias,
        #     # scale=self.scale
        # )

        # # Reshape back to (batch_size, seq_length, embed_dim)
        # attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        
        attn_outputs = []
        for i in range(batch_size):
            seq_len = int(mask[i].sum().item())
            
            if seq_len == 0:
                attn_outputs.append(torch.zeros((seq_length, embed_dim), device=Q.device, dtype=Q.dtype))
                continue

            single_query = Q[i, :seq_len].unsqueeze(0)
            single_key = K[i, :seq_len].unsqueeze(0)
            single_value = V[i, :seq_len].unsqueeze(0)
            
            if self.verbosity > 1:
                print(f"Batch {i}: seq_len = {seq_len}, Q.shape = {single_query.shape}")
            attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens([seq_len]).to(Q.device)

            # Compute memory-efficient attention for a single sequence
            single_attn_output = xops.memory_efficient_attention(
                query=single_query,
                key=single_key,
                value=single_value,
                attn_bias=attn_bias,
                scale=self.scale
            )
            single_attn_output = single_attn_output.transpose(1, 2).reshape(seq_len, embed_dim)

            padded_output = torch.zeros((seq_length, embed_dim), device=Q.device, dtype=Q.dtype)
            padded_output[:seq_len] = single_attn_output
            attn_outputs.append(padded_output)

        # Concatenate outputs back into the batch dimension
        attn_output = torch.stack(attn_outputs, dim=0)
        if self.nan_logger:
            self.nan_logger.info(f"---------- attention(xFormers) ---------- ")
            self.nan_logger.info(f"attn_output has NaN: {torch.isnan(attn_output).any()}")

        return self.out_proj(attn_output)
    
    # def _expand_mask_for_bias(self, mask, batch_size, seq_length):
    #     """
    #     1. mask(arg) shape: [batch_size, seq_length]
    #     2. mask[:, None, None, :] shape: [batch_size, 1, 1, seq_length]
    #     3. mask[:, None, :, None] shape: [batch_size, 1, seq_length, 1]
    #     4. 2*3: [batch_size, 1, seq_length, seq_length]
    #     5. expanded_mask shape: [batch_size, n_heads, seq_length, seq_length]
    #     """
    #     XFORMER_ALIGN_CONSTANT = 8
    #     aligned_seq_length = ((seq_length + XFORMER_ALIGN_CONSTANT - 1) // XFORMER_ALIGN_CONSTANT) * XFORMER_ALIGN_CONSTANT
    #     print(f" aligned_seq_length: {aligned_seq_length}")
        
    #     aligned_mask = torch.zeros((batch_size, self.n_heads, aligned_seq_length, aligned_seq_length), dtype=torch.float32, device=mask.device)
    #     expanded_mask = mask[:, None, None, :] * mask[:, None, :, None]
    #     expanded_mask = expanded_mask.expand(-1, self.n_heads, -1, -1)
        
    #     expanded_mask = expanded_mask.masked_fill(expanded_mask == 0, float("-inf"))
        
    #     aligned_mask[:, :, :seq_length, :seq_length] = expanded_mask
    #     print(f" aligned_mask shape: {aligned_mask.shape}")
    #     return aligned_mask[:, :, :seq_length, :seq_length]

