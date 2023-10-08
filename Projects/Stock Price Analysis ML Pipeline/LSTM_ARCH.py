# -*- coding: utf-8 -*-
"""
Architecture for the LSTM model ---
"""
import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=16, num_layers=1)
        self.linear = nn.Linear(16,1)
      
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
