import torch
import torch.nn as nn

class Pooling(nn.Module):
    def __init__(self, pooling_type="mean"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
            x = x * mask  # Zero out padding

            if self.pooling_type == "mean":
                return x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # Avoid division by zero
            elif self.pooling_type == "max":
                return (x + (mask - 1) * -1e9).max(dim=1)[0]  # Large negative value for padding
            elif self.pooling_type == "synthetic":
                mean_pooled = x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
                max_pooled, _ = (x + (mask - 1) * -1e9).max(dim=1)
                return torch.cat([mean_pooled, max_pooled], dim=-1)  # Concatenate mean and max pooling
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Fallback to default pooling if no mask provided
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        elif self.pooling_type == "synthetic":
            return torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
