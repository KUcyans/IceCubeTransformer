import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_f, dropout=0.01,):
        super().__init__()
        self.d_model = d_model
        self.d_f = d_f
        self.W_h = nn.Linear(self.d_model, self.d_f)
        self.W_f = nn.Linear(self.d_f, self.d_model)
        
        # self.activation = nn.ReLU()
        self.activation = nn.GELU()
        # self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # (batch_size, seq_length, d_f)
        s_i = self.W_h(x)# summed input
        # (batch_size, seq_length, d_f)
        h_i = self.activation(s_i)
        h_i = self.dropout(h_i)
        
        # (batch_size, seq_length, d_model)
        x_next = self.W_f(h_i) # summed output
        
        return x_next