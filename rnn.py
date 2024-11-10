import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pred_length):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            nonlinearity="relu",
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.mlp = nn.Linear(hidden_dim, output_dim * pred_length)

        self.create_weights()

    def create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.mlp(out[:, -1, :])

        return out
