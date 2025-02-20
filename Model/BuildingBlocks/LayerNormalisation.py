import torch
import torch.nn as nn

class LayerNormalisation(nn.Module):
    def __init__(self, d_n, eps=1e-6):
        # d_n (int): dimension of the normalisation layer, 
        # here d_n = d_model = embed_dim
        # eps: epsilon, a small number to avoid division by zero
        super().__init__()
        self.d_n = d_n
        self.eps = eps

        self.g = nn.Parameter(torch.ones(d_n)) # gain
        self.b = nn.Parameter(torch.zeros(d_n)) # bias

    def forward(self, x):
        # (batch_size, seq_length, 1)
        mu = x.mean(dim=-1, keepdim=True)

        # (batch_size, seq_length, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normalised = (x - mu) / torch.sqrt(var + self.eps)

        # (batch_size, seq_length, d_n)
        x_tilde = self.g * x_normalised + self.b
        
        return x_tilde