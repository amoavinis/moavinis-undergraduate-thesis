import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

""" This class instantiates an LSTM module. """
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=1, mode=None, device='cpu'):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            num_layers)    

        self.hidden_cell = (nn.Parameter(torch.zeros(self.num_layers,
                                                     self.output_dim,
                                                     self.hidden_dim)),
                            nn.Parameter(torch.zeros(self.num_layers,
                                                     self.output_dim,
                                                     self.hidden_dim)))


    def forward(self, x):        
        h, c = self.hidden_cell
        h = h.detach().to(self.device)
        c = c.detach().to(self.device)
        x, self.hidden_cell = self.lstm(x, (h, c))

        return x
        
