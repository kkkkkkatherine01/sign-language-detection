# Training

One script per model in [`src/models`](../models): `train_lstm.py`,
`train_tcn.py`, `train_transformer.py`. All three share the same loop:
`BCEWithLogitsLoss` with `pos_weight` computed automatically from the
training split's class imbalance, `AdamW` + `ReduceLROnPlateau` (on val
accuracy), and early stopping (`--patience`).

Every script needs exactly one data source:
`--data_dir /path/to/merged` (output of
[`merge_datasets.py`](../preprocessing/merge_datasets.py)) **or**
`--h5_path /path/to/dataset.h5` (output of
[`../utils/pack_to_h5.py`](../utils/pack_to_h5.py)).

## LSTM

```bash
python train_lstm.py \
  --data_dir /path/to/merged --save_dir runs/lstm_bh \
  --input_size 75 --hidden_size 64 --num_layers 1 --epochs 100
```
Add `--bidirectional` for a BiLSTM.

## TCN

```bash
python train_tcn.py \
  --data_dir /path/to/merged --save_dir runs/tcn_bh \
  --input_size 75 --n_channels 48 --kernel_size 5 --dilations 1 2 4 8 16
```

## Transformer

```bash
python train_transformer.py \
  --data_dir /path/to/merged --save_dir runs/transformer_bh \
  --input_size 75 --d_model 64 --nhead 4 --num_encoder_layers 2 --window_size 250
```
Only this script supports `--num_gpus N` (`DataParallel`) and automatically
resumes from `checkpoint_latest.pt` in `--save_dir` if one exists — LSTM/TCN
have no resume logic, re-running them starts from scratch.

For `_face` features (167-dim) on any of the three, pass `--input_size 167`.

## Produces

Written to `--save_dir` as training runs:
```
args.json               # all CLI args as passed — evaluate.py reads this back
history.json            # per-epoch train/val loss + accuracy
best_model.pt           # state_dict of the best-val-accuracy checkpoint
checkpoint_latest.pt     # (transformer only) full resume state — model/optimizer/scheduler/history
```

The console prints per-epoch `train/val loss + acc` and a final test-set
loss/accuracy line at the end. That's a quick sanity check only — run
[`src/evaluation/evaluate.py`](../evaluation) against `--save_dir` for the
full classification report, ROC curve, and per-dataset breakdown.

## Other useful flags (all three scripts)

- `--epochs`, `--batch_size`, `--val_batch_size`, `--lr`, `--weight_decay`,
  `--patience`, `--seed`, `--num_workers` — standard training knobs.
- `--steps_per_epoch N` — batches per epoch (`-1` = one full pass over the
  training set each epoch; LSTM defaults to `128`, TCN/Transformer default
  to `-1`).
