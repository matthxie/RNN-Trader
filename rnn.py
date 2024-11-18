import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, pred_length, dropout
    ):
        super(RNN, self).__init__()

        self.pred_length = pred_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim * pred_length)

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])
        out = out.view(-1, self.pred_length, self.output_dim)

        return out
