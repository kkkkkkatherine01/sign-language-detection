# Evaluation

`evaluate.py` runs a trained model on the test split and writes out metrics
+ plots. Model hyperparameters are read back from the `args.json` that
`train_*.py` saved, so you only need `--model_type` and `--save_dir`
pointing at that training run's checkpoint folder.

```bash
python evaluate.py --model_type lstm \
  --data_dir /path/to/merged --save_dir runs/lstm_bh

python evaluate.py --model_type tcn \
  --h5_path /path/to/dataset.h5 --save_dir runs/tcn_bh

python evaluate.py --model_type transformer \
  --data_dir /path/to/merged --save_dir runs/transformer_bh
```
`--save_dir` must be the same folder passed to the matching `train_*.py`
(it needs `args.json` + `best_model.pt` from that run). `--data_dir` /
`--h5_path` don't have to be the exact same flag used for training — just
the same underlying test split.

## Produces

Written to `--save_dir`:
```
test_predictions.npz          # per-frame probs/labels/preds/sources/seq_ids
classification_report.txt     # precision/recall/F1, Non-signing vs Signing

plots/training_curves.png     # from history.json, if present
plots/confusion_matrix.png
plots/roc_curve.png
plots/prob_distribution.png    # predicted-probability histograms, per class
plots/per_dataset_accuracy.png # accuracy broken out by meta "source" (dgs/how2sign/ted/...)
```
Console also prints the classification report and the ROC AUC.

`test_predictions.npz` is the input to
[`../utils/bootstrap_eval.py`](../utils/bootstrap_eval.py) for bootstrap
confidence intervals on test accuracy:
```bash
python ../utils/bootstrap_eval.py --pred_path runs/lstm_bh/test_predictions.npz --n_boot 1000
```

## Other flags

- `--threshold` (default `0.5`) — decision threshold applied to the sigmoid
  output when turning probabilities into 0/1 predictions.
- `--batch_size`, `--num_workers` — inference-time DataLoader settings.
