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
        self.out_dim = out_dim
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
# --- TCN REGRESSOR -----------------------------------------------------------
# Follows the common TCN structure: stacks of residual blocks with causal,
# dilated Conv1d layers (weight_norm) and dropout. Input: (B, T, F),
# Conv1d expects (B, C_in, T). We use causal padding so that output at t
# only depends on <= t. We predict the last time step: y_hat = head(h[:, :, -1]).
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """Remove the trailing padding so convolution is strictly causal."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding on the left
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs,
                                           kernel_size,
                                           padding=pad,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs,
                                           kernel_size,
                                           padding=pad,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.final_relu = nn.ReLU()

        # Kaiming init is fine for ReLU; weight_norm handles scale during training.
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="linear")

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(y + res)

class TCNBackbone(nn.Module):
    def __init__(self, in_channels, channels_list, kernel_size, dropout):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels_list):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size, dilation, dropout))
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C, T)
        return self.net(x)  # (B, C_last, T)

class TCNRegressor(nn.Module):
    """
    Sequence-to-one regressor:
      input  xb: (B, T, F)
      output yh: (B, out_dim)  (last time step representation)
    """
    def __init__(self,
                 input_size: int,
                 out_dim: int,
                 channels_list=(64, 64, 64),
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        assert kernel_size >= 2 and kernel_size % 1 == 0
        self.backbone = TCNBackbone(in_channels=input_size,
                                    channels_list=channels_list,
                                    kernel_size=kernel_size,
                                    dropout=dropout)
        self.head = nn.Linear(channels_list[-1], out_dim)

    def forward(self, xb):  # xb: (B, T, F)
        x = xb.transpose(1, 2)         # (B, F, T)
        h = self.backbone(x)           # (B, C, T)
        h_last = h[:, :, -1]           # last time step (causal)
        return self.head(h_last)       # (B, out_dim)
