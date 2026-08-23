"""
Per-frame LSTM sign language detection model.
Uses pack_padded_sequence to skip padding frames.
Input:  (batch, T, input_size) + lengths
Output: (batch, T) logit per frame
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMDetectorPerFrame(nn.Module):
    def __init__(
        self,
        input_size: int = 75,
        hidden_size: int = 64,
        num_layers: int = 1,
        input_dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_norm    = nn.LayerNorm(input_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # lengths must live on CPU for pack_padded_sequence
        x = self.input_norm(x)
        x = self.input_dropout(x).contiguous()

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        logits = self.classifier(out)
        return logits.squeeze(-1)
