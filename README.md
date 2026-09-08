# Sign Language Detection

Per-frame binary detection of "is the person signing right now?" from
MediaPipe body/hand(+face) keypoint sequences, comparing three sequence
models (LSTM, TCN, sliding-window Transformer) trained on a mix of sign
language and non-signing video sources.

## Pipeline

```
keypoint JSON (MediaPipe)
        │
        ▼
src/preprocessing/   process_*.py → per-dataset feature/label .npy
                      merge_datasets.py → merged train/val/test
        │
        ▼
src/utils/           pack_to_h5.py (optional: pack merged .npy into one .h5)
        │
        ▼
src/training/        train_lstm.py / train_tcn.py / train_transformer.py
        │             (uses model classes + Dataset from src/models/)
        ▼
src/evaluation/      evaluate.py → classification report, ROC, plots
        │
        ▼
src/utils/           bootstrap_eval.py (bootstrap CI on test accuracy)
```

Each folder has its own README with full command reference:

- [`src/preprocessing/`](src/preprocessing/README.md) — per-dataset keypoint
  → feature/label conversion, then merging into a single train/val/test set
- [`src/models/`](src/models/README.md) — model architectures and the
  Dataset/DataLoader code (library code, not run directly)
- [`src/training/`](src/training/README.md) — how to train each model
- [`src/evaluation/`](src/evaluation/README.md) — how to evaluate a
  checkpoint and read the outputs

## Quickstart

```bash
# 1. preprocess each dataset (repeat per dataset, see src/preprocessing/README.md)
python src/preprocessing/process_dgs.py --dgs_dir_a ... --eaf_dir ... --output_dir intermediate/
python src/preprocessing/process_how2sign.py --train ... --val ... --test ... --output_dir intermediate/
python src/preprocessing/process_ted.py --ted_dir ... --output_dir intermediate/

# 2. merge into one train/val/test set
python src/preprocessing/merge_datasets.py \
  --input_dir intermediate/ --datasets how2sign dgs ted --output_dir merged/ --shuffle

# 3. train
python src/training/train_lstm.py --data_dir merged/ --save_dir runs/lstm_bh

# 4. evaluate
python src/evaluation/evaluate.py --model_type lstm --data_dir merged/ --save_dir runs/lstm_bh
```

## Datasets

| Dataset | Role | Labels |
|---|---|---|
| DGS Corpus | train/val/test | sentence-level, ELAN `.eaf` annotations |
| How2Sign | train/val/test | every frame = signing (positive-only source) |
| TED Talks | train/val/test | every frame = non-signing (negative-only source) |
| SSLC | evaluation only | sentence-level, ELAN `.eaf` annotations |

## Setup

```bash
pip install -r requirements.txt
```
Install the CUDA build of PyTorch matching your GPU/driver if training on
GPU — see the comment in [`requirements.txt`](requirements.txt).

## Repo structure

```
src/
  preprocessing/   keypoint JSON → per-dataset features/labels → merged dataset
  models/          model classes (LSTM / TCN / Transformer) + Dataset/DataLoader
  training/        train_lstm.py, train_tcn.py, train_transformer.py
  evaluation/      evaluate.py
  utils/           pack_to_h5.py, bootstrap_eval.py
```

## License

[MIT](LICENSE) — the code only. The datasets referenced above each have
their own license/usage terms and are not included in this repo.
