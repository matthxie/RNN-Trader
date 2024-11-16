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
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, nonlinearity="relu", batch_first=True
        )
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim * pred_length)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # out = self.linear1(x)
        # out = self.batch_norm(out)
        # out = self.relu(out)
        out, _ = self.gru(x)
        out = self.dropout(out)
        # out = out.view(-1, self.output_dim)
        out = self.linear2(out[:, -1, :])
        out = out.view(-1, self.pred_length, self.output_dim)

        return out
