"""
model_tcn_perframe.py

Per-frame TCN (Temporal Convolutional Network) sign language detection model.
Reference: Bai et al. (2018), "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling".

A 1x1 Conv1d projects the input into 5 causal residual blocks (dilations
1/2/4/8/16, each two dilated Conv1d + ReLU + dropout with a skip connection),
then a Linear classifier per frame. Causal = left-padded only, so the
prediction at time t only sees t, t-1, t-2, ... — same constraint as the
single-directional LSTM baseline.

Receptive field = 1 + (kernel-1) * sum(dilations) = 63 frames, ~2.5s at 25fps.
Params: ~33K at input_size=75 vs ~36K for the LSTM on the same features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNResidualBlock(nn.Module):
    """
    One causal TCN residual block: two causal dilated Conv1d layers + skip connection.

    Causal padding: pad (kernel-1)*dilation frames on the LEFT only.
    This ensures output at time t only sees input at times <= t.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.causal_pad = dilation * (kernel_size - 1)

        self.conv1 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.conv2 = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(x, (self.causal_pad, 0))
        out = self.dropout(self.relu(self.conv1(out)))

        out = F.pad(out, (self.causal_pad, 0))
        out = self.dropout(self.relu(self.conv2(out)))

        return self.relu(out + x)


class TCNDetectorPerFrame(nn.Module):
    def __init__(
        self,
        input_size: int = 75,
        n_channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple = (1, 2, 4, 8, 16),
        input_dropout: float = 0.5,
        block_dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)
        self.input_proj = nn.Conv1d(input_size, n_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(n_channels, kernel_size, d, block_dropout)
            for d in dilations
        ])
        self.classifier = nn.Linear(n_channels, 1)

        rf = 1 + (kernel_size - 1) * sum(dilations)
        print(f"[TCN] causal receptive field = {rf} frames "
              f"(≈{rf/25:.1f}s at 25fps) | "
              f"dilations={list(dilations)} | kernel={kernel_size}")

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        # lengths is unused — kept so this drops in for the LSTM model
        x = self.input_dropout(x)
        x = x.transpose(1, 2)              # (B, F, T) for Conv1d

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        logits = self.classifier(x)
        return logits.squeeze(-1)
