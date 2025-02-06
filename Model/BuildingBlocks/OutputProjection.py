import torch
import torch.nn as nn

class OutputProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Reduce dimensionality
            nn.ReLU(),  # Non-linearity
            nn.Dropout(p=dropout),  # Optional dropout
            nn.Linear(hidden_dim, output_dim)  # Final classification
        )

    def forward(self, x):
        return self.projection(x)