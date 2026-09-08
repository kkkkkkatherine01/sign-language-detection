# Models

Model definitions and the shared `Dataset`/`DataLoader` code used by
[`src/training`](../training) and [`src/evaluation`](../evaluation). Nothing
in this folder is run directly — no `argparse`/`main`, just classes imported
elsewhere.

All three models implement the same interface: `forward(x, lengths)` takes
padded input `(B, T, input_size)` and returns per-frame logits `(B, T)`
(pre-sigmoid; paired with `BCEWithLogitsLoss` in training).

## `dataset.py`

Two interchangeable data sources, selected by the training/evaluation
scripts' `--data_dir` vs `--h5_path` flag:

- **`SignSequenceDataset(data_dir, split)`** — loads
  `{split}_sequences.npy` / `{split}_labels.npy` (the merged output of
  [`merge_datasets.py`](../preprocessing/merge_datasets.py)). `__getitem__`
  returns one `(X, y)` pair of float32/float32 tensors, full sequence length.
- **`H5SignSequenceDataset(h5_path, split)`** — same interface, reads from a
  packed `.h5` file instead (output of
  [`../utils/pack_to_h5.py`](../utils/pack_to_h5.py)). Opens its own file
  handle per DataLoader worker.

`collate_fn(batch)` pads a batch of variable-length sequences to the batch's
max length and returns `(X_pad, y_pad, mask, lengths)`, where `mask` is
`True` on real (non-padded) frames — training/eval scripts index with `mask`
before computing loss/accuracy so padding never affects metrics.

`get_dataloaders(data_dir, batch_size, num_workers)` /
`get_dataloaders_h5(h5_path, batch_size, num_workers)` build train/val/test
`DataLoader`s (train shuffled, val/test not) in one call and return
`(train_loader, val_loader, test_loader, test_dataset)`.

## `model_lstm.py` — `LSTMDetectorPerFrame`

`LayerNorm` → input dropout → `nn.LSTM` (packed via `pack_padded_sequence`,
optionally bidirectional) → linear classifier on every timestep.

```python
from model_lstm import LSTMDetectorPerFrame
model = LSTMDetectorPerFrame(input_size=75, hidden_size=64, num_layers=1,
                              bidirectional=False)
logits = model(x, lengths)   # x: (B, T, 75) padded, lengths: (B,) on CPU
```

## `model_tcn.py` — `TCNDetectorPerFrame`

1×1 `Conv1d` input projection → stack of causal dilated-`Conv1d` residual
blocks (`TCNResidualBlock`, each left-padded so no future frame leaks in) →
linear classifier. `lengths` is accepted for interface parity but unused —
convolutions run over the full padded tensor; masking happens downstream.

```python
from model_tcn import TCNDetectorPerFrame
model = TCNDetectorPerFrame(input_size=75, n_channels=32, kernel_size=3,
                             dilations=(1, 2, 4, 8, 16))
logits = model(x, lengths)   # lengths ignored, kept for a shared call signature
```

## `model_transformer.py` — `TransformerDetectorPerFrame`

Encoder-only Transformer, but attention is computed over a **sliding causal
window** (`window_size` frames, 50% hop/overlap) instead of the full
sequence — keeps compute at `O(T·W)` instead of `O(T²)` for long videos.
Sinusoidal positional encoding, `norm_first` pre-LN encoder layers.

```python
from model_transformer import TransformerDetectorPerFrame
model = TransformerDetectorPerFrame(input_size=75, d_model=64, nhead=4,
                                     num_encoder_layers=2, window_size=250)
logits = model(x, lengths)   # lengths used for both padding & causal masks
```

`d_model` must be divisible by `nhead`; `window_size >= 2`.

## Feature dimensions

`input_size` must match what preprocessing produced: 75 (body+hand,
optical_flow) / 150 (raw_xy), or 167 / 334 with `_face` features — see
[`src/preprocessing`](../preprocessing).
