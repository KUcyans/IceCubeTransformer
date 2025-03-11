import torch
from torch import nn

class Pooling(nn.Module):
    def __init__(self, pooling_type="mean"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x, mask=None):
        if mask is not None:
            # x shape is (batch_size, seq_len, d_model)
            mask = mask.unsqueeze(-1) # Expands to (batch_size, seq_len, 1) to match x
            x = x * mask  

            if self.pooling_type == "mean":
                # Sum along the sequence dimension and normalize by mask (event length)
                return x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # Prevent division by zero
            elif self.pooling_type == "max":
                # Apply max pooling with mask
                return (x + (mask - 1) * -1e9).max(dim=1)[0]
            elif self.pooling_type == "synthetic":
                # Concatenate mean and max pooling
                mean_pooled = x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
                max_pooled, _ = (x + (mask - 1) * -1e9).max(dim=1)
                return torch.cat([mean_pooled, max_pooled], dim=-1)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Default case if no mask is provided
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        elif self.pooling_type == "synthetic":
            return torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
