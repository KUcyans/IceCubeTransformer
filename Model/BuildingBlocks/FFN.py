import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_f, dropout=0.1, nan_logger=None):
        super().__init__()
        self.d_model = d_model
        self.d_f = d_f
        self.W_h = nn.Linear(self.d_model, self.d_f)
        self.W_f = nn.Linear(self.d_f, self.d_model)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.nan_logger = nan_logger
        
    def forward(self, x):
        # (batch_size, seq_length, d_f)
        s_i = self.W_h(x)# summed input
        # (batch_size, seq_length, d_f)
        h_i = self.activation(s_i)
        h_i = self.dropout(h_i)
        
        # (batch_size, seq_length, d_model)
        x_next = self.W_f(h_i) # summed output
        
        self.nan_logger.info(f"---------FFN-----------")
        self.nan_logger.info(f"s_i hasn't nan : {not torch.isnan(s_i).any()}")
        self.nan_logger.info(f"h_i hasm't nan : {not torch.isnan(h_i).any()}")
        self.nan_logger.info(f"x_next hasn't nan : {not torch.isnan(x_next).any()}")
        
        return x_next