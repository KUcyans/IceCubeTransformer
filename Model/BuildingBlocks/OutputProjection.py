import torch.nn as nn

class OutputProjection(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_f: int,
                 num_classes: int,
                 num_layers: int,
                 dropout=0.01):
        super().__init__()
        layers = []
        in_dim = d_model

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, d_f))
            # layers.append(nn.ReLU())
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = d_f
        layers.append(nn.LayerNorm(d_f))
        layers.append(nn.Linear(d_f, num_classes))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)