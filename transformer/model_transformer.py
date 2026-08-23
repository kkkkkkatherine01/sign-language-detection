"""
Per-frame Transformer sign language detection model. Drop-in replacement
for LSTMDetectorPerFrame — same forward signature.

Uses a sliding causal attention window (50% overlap) instead of full
self-attention, so it handles long sequences without O(T^2) memory and
stays real-time-streaming compatible, same as the LSTM baseline.

Input:  (batch, T, input_size) + lengths
Output: (batch, T) logit per frame
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017), no learnable
    parameters. max_len=200000 matches Saunders et al.
    """

    def __init__(self, d_model: int, max_len: int = 200000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (batch, T, d_model)
        # offset: global start position of this chunk
        return x + self.pe[:, offset : offset + x.size(1)]


class TransformerDetectorPerFrame(nn.Module):
    """
    Encoder-only Transformer with sliding window attention.

    window_size (W): number of frames per attention window (default 50 = 2s @25fps)
    overlap      : W//2 frames shared between consecutive windows
    """

    def __init__(
        self,
        input_size:         int   = 75,
        d_model:            int   = 64,
        nhead:              int   = 4,
        num_encoder_layers: int   = 2,
        dim_feedforward:    int   = 256,
        dropout:            float = 0.1,
        input_dropout:      float = 0.5,
        window_size:        int   = 50,
    ):
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        assert window_size >= 2, "window_size must be at least 2"

        self.window_size   = window_size
        self.input_norm    = nn.LayerNorm(input_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_proj    = nn.Linear(input_size, d_model)
        self.pos_enc       = PositionalEncoding(d_model)
        self.emb_dropout   = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
            layer_norm_eps  = 1e-6,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = num_encoder_layers,
            norm                 = nn.LayerNorm(d_model, eps=1e-6),
            enable_nested_tensor = False,
        )
        self.classifier = nn.Linear(d_model, 1)

    # ------------------------------------------------------------------
    def _make_padding_mask(self, lengths: torch.Tensor, max_T: int) -> torch.Tensor:
        """(batch, T) bool mask: True = padding (ignored by attention)."""
        device = lengths.device
        return torch.arange(max_T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

    # ------------------------------------------------------------------
    def _encode_window(self, x_win, lengths_win):
        W = x_win.size(1)

        valid_mask = ~self._make_padding_mask(lengths_win, W)
        x_win = x_win * valid_mask.unsqueeze(-1).float()

        # rows that fall entirely in padding get an unmasked key set —
        # an all-True padding mask plus the causal mask leaves nothing to
        # attend to, which produces NaN in the backward pass on ROCm
        key_padding_mask = self._make_padding_mask(lengths_win, W)
        if lengths_win.min() == 0:
            key_padding_mask[lengths_win == 0] = False

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            W, device=x_win.device
        )

        return self.transformer_encoder(
            x_win,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        W       = self.window_size
        hop     = W // 2
        device  = x.device
        lengths = lengths.to(device)

        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = self.input_proj(x)

        out_buffer = torch.zeros(B, T, x.size(-1), device=device)

        # Slide a causal window across the sequence with 50% overlap; only
        # the non-overlapping tail of each window is kept (the first window
        # keeps everything), so every kept frame has `hop` frames of context.
        start = 0
        first_window = True
        while start < T:
            end = min(start + W, T)
            x_win = x[:, start:end]

            x_win = self.pos_enc(x_win, offset=0)
            x_win = self.emb_dropout(x_win)

            lengths_win = (lengths - start).clamp(min=0, max=end - start)
            out_win = self._encode_window(x_win, lengths_win)

            if first_window:
                keep_start = 0
                first_window = False
            else:
                keep_start = hop

            keep_global_start = start + keep_start
            keep_global_end   = start + (end - start)
            out_buffer[:, keep_global_start:keep_global_end] = \
                out_win[:, keep_start:]

            if end == T:
                break
            start += hop

        logits = self.classifier(out_buffer)
        return logits.squeeze(-1)
