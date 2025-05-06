import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional # Import Optional for type hinting

class AttentionHeadBase(nn.Module, ABC):
    """
    Abstract Base Class for core attention mechanisms operating on multi-head tensors.

    These mechanisms take Q, K, V tensors already split by heads 
    (shape: B, H, S, D) and compute the attention output, also of 
    shape (B, H, S, D).
    """
    def __init__(self, 
                 head_dim: int, 
                 n_heads: int,
                 dropout: float = 0.01):
        """
        Initializes common attributes.

        Args:
            head_dim (int): Dimension of each attention head.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability. Note: Subclasses determine *where* dropout is applied (e.g., on weights vs using F.sdpa dropout).
        """
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        # Store the dropout probability; subclasses decide how/where to use it
        # This is often more flexible than creating the nn.Dropout module here.
        self.dropout_p = dropout 

    @abstractmethod
    def forward(self, 
                q: torch.Tensor, #(batch_size, n_heads, seq_len_q, head_dim)
                k: torch.Tensor, #(batch_size, n_heads, seq_len_k, head_dim)
                v: torch.Tensor, #(batch_size, n_heads, seq_len_k, head_dim)
                event_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the core attention computation.

        Args:
            q (torch.Tensor): Queries tensor, shape (batch_size, n_heads, seq_len_q, head_dim)
            k (torch.Tensor): Keys tensor, shape (batch_size, n_heads, seq_len_k, head_dim)
            v (torch.Tensor): Values tensor, shape (batch_size, n_heads, seq_len_k, head_dim)
            event_length (Optional[torch.Tensor]): Optional tensor indicating the true sequence 
                                                   length for each item in the batch, used for masking.
                                                   Expected shape (batch_size,).

        Returns:
            torch.Tensor: Attention output tensor, shape (batch_size, n_heads, seq_len_q, head_dim)
        """
        pass
    
    @staticmethod
    def make_attention_mask(batch_event_length, max_len):
        """
        Constructs a binary mask for attention based on per-batch event lengths.

        Args:
            batch_event_length (Tensor): shape (B,), actual lengths
            max_len (int): maximum length (typically sequence length)

        Returns:
            Tensor: shape (B, 1, 1, max_len), where True indicates keep.
        """
        mask = torch.arange(max_len, device=batch_event_length.device).expand(len(batch_event_length), max_len)
        mask = mask < batch_event_length.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(2)