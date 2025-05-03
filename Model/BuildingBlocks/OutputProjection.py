import torch
import torch.nn as nn

class OutputProjection(nn.Module):
    def __init__(self, d_model, d_f, num_classes, dropout=0.01):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_f),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_f, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        return self.projection(x)