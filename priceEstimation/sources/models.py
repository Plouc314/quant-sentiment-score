import torch
import torch.nn as nn

############## LSTM : time series

class StockLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x) # initial h0, c0 --> default zero
        
        out = self.fc(out[:, -1, :]) 
        return out


############## Transformer : encoder decoder

############## Foundation model