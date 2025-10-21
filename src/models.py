import torch
import torch.nn as nn

class ConvFrontEnd(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
    def forward(self, x):
        # x: (B, T, F) → (B, F, T) → conv → (B, T, C)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=1, dropout=0.2, rnn_type='lstm', conv_frontend=True, conv_channels=64, out_dim=3):
        super().__init__()
        self.conv_frontend = ConvFrontEnd(input_size, conv_channels) if conv_frontend else None
        rnn_input = conv_channels if conv_frontend else input_size
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers>1 else 0.0))
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(rnn_input, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers>1 else 0.0))
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        # x: (B, T, F)
        if self.conv_frontend is not None:
            x = self.conv_frontend(x)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.drop(last)
        y = self.head(last)
        return y
