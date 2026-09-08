# Preprocessing

Turns per-frame MediaPipe keypoint JSON (extracted upstream, not part of this
repo) into the feature/label arrays consumed by [`src/models`](../models) and
[`src/training`](../training). Each script handles one dataset; the `_face`
variant (`process_dgs_face.py`, etc.) does the same thing but also reads a
matching face-keypoint JSON and adds face landmarks to the features. Shared
code lives in [`preprocessing_utils.py`](preprocessing_utils.py).

Add `--feature_type raw_xy` to any command below to get flattened normalized
coordinates instead of the default optical-flow features (frame-to-frame
motion). Dimensions: 75 (body+hand, optical_flow) / 150 (raw_xy), or 167 /
334 with `_face`.

## DGS Corpus

```bash
python process_dgs.py \
  --dgs_dir_a /path/to/keypoints/signerA --dgs_dir_b /path/to/keypoints/signerB \
  --eaf_dir /path/to/eaf --output_dir out/

# body+hand+face:
python process_dgs_face.py \
  --dgs_dir_a /path/to/keypoints/signerA --dgs_face_dir_a /path/to/face/signerA \
  --eaf_dir /path/to/eaf --output_dir out/
```
`--dgs_dir_b` / `--dgs_face_dir_b` are optional (second signer). Videos
without a matching `.eaf` are skipped entirely. Sentence-level labels come
from the `Deutsche_Übersetzung_{signer}` ELAN tier. Input is 50 fps and gets
downsampled to 25 fps automatically.

**Produces**, in `out/`:
```
dgs_split_map.json
dgs_train_sequences.npy   dgs_train_labels.npy   dgs_train_meta.json
dgs_val_sequences.npy     dgs_val_labels.npy     dgs_val_meta.json
dgs_test_sequences.npy    dgs_test_labels.npy    dgs_test_meta.json
```
(`dgs_face_*` for the `_face` script.)

## How2Sign

```bash
python process_how2sign.py \
  --train /path/to/train_kp --val /path/to/val_kp --test /path/to/test_kp \
  --output_dir out/

# body+hand+face:
python process_how2sign_face.py \
  --train /path/to/train_kp --face_train /path/to/train_face_kp \
  --val /path/to/val_kp --face_val /path/to/val_face_kp \
  --test /path/to/test_kp --face_test /path/to/test_face_kp \
  --output_dir out/
```
Directories are already pre-split train/val/test. Every frame is labeled `1`
(signing) — this dataset only supplies positive examples.

**Produces**:
```
how2sign_train_sequences.npy  how2sign_train_labels.npy  how2sign_train_meta.json
how2sign_val_*                how2sign_test_*
```
(`how2sign_face_*` for the `_face` script.)

## SSLC

```bash
python process_sslc.py \
  --pairs_file sslc_available_pairs.txt --keypoint_dir /path/to/keypoints \
  --eaf_dir /path/to/eaf --output_dir out/

# body+hand+face:
python process_sslc_face.py \
  --pairs_file sslc_available_pairs.txt \
  --keypoint_dir /path/to/keypoints --face_keypoint_dir /path/to/face_keypoints \
  --eaf_dir /path/to/eaf --output_dir out/
```
`--pairs_file` is a tab-separated file mapping `video_file` → `signer` →
`eaf_file`. No train/val/test split — this dataset is evaluation-only.

**Produces**:
```
sslc_sequences.npy   sslc_labels.npy   sslc_meta.json
```
(`sslc_face_*` for the `_face` script.)

## TED Talks

```bash
python process_ted.py --ted_dir /path/to/keypoints --output_dir out/

# body+hand+face:
python process_ted_face.py \
  --ted_dir /path/to/keypoints --face_dir /path/to/face_keypoints \
  --output_dir out/
```
Every frame is labeled `0` (non-signing) — negative-class source. Frame
transitions that look like camera cuts (large motion **and** a large
shoulder-position jump at the same time) are filtered out automatically.

**Produces**:
```
ted_split_map.json
ted_train_sequences.npy  ted_train_labels.npy  ted_train_meta.json
ted_val_*                ted_test_*
```
(`ted_face_*` for the `_face` script, plus its own `ted_face_split_map.json`.)

## Merging datasets

Each script above writes its own `{dataset}_{split}_*` files. Before training,
combine the datasets you want into a single `train`/`val`/`test` set with
[`merge_datasets.py`](merge_datasets.py):

```bash
python merge_datasets.py \
  --input_dir out/ --datasets how2sign dgs ted --output_dir merged/ \
  --shuffle
```
`--datasets` controls which ones go in and in what order they're
concatenated; `--shuffle` randomizes sequence order within each split
afterwards (off by default). SSLC has no train/val/test split, so it's meant
for standalone evaluation and is not passed to `--datasets` here.

**Produces**, in `merged/`:
```
train_sequences.npy  train_labels.npy  train_meta.json
val_*                test_*
*_split_map.json     (copied over from --input_dir, one per split-based dataset)
```

This merged output is what `src/models`/`src/training` expect via
`--data_dir`. Optionally, pack it into a single `.h5` file with
[`../utils/pack_to_h5.py`](../utils/pack_to_h5.py) to hand training a single
file instead of six.

## Output array shapes

- `*_sequences.npy` — object array, one entry per video, each a
  `(T_i, D)` float32 array (`D` = 75/150/167/334 depending on
  feature_type/face).
- `*_labels.npy` — object array, one entry per video, each a `(T_i,)` int8
  array of 0/1 labels aligned frame-for-frame with the sequence.
- `*_meta.json` — list of dicts (source, fps, signing_ratio, ...), one per
  video, same order as the arrays.

Sequences are variable-length, not padded — batching/padding happens in
[`src/models/dataset.py`](../models/dataset.py).
