"""TCN  sign language detection model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNResidualBlock(nn.Module):
    """
    One causal TCN residual block: two causal dilated Conv1d layers + skip connection.
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

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x = self.input_dropout(x)
        x = x.transpose(1, 2)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        logits = self.classifier(x)
        return logits.squeeze(-1)
