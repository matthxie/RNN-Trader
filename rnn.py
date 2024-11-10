import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, nonlinearity='relu', batch_first=False, dropout=0.0, bidirectional=False)
        self.mlp = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if isinstance(x, tuple):  # Prevent tuple input error
            raise ValueError("Input should be a tensor, not a tuple.")

        # h0 = Variable(torch.zeros(self.num_layers, 5, self.hidden_dim))

        out, _ = self.rnn(x)
        out = self.mlp(out[-1, :])

        return out
    
