"""
process_how2sign_face.py

Preprocess How2Sign keypoint JSON files (body+hand+face) into per-frame
features and labels.

All How2Sign frames are labelled signing=1.
No outlier filtering is applied.

Body and face JSONs must share the same filename; only videos that have
both a body and a face JSON are processed.

Feature dims:
  optical_flow: 167  (75 body+hand + 92 face, L2 norm per keypoint)
  raw_xy:       334  (167 × 2, xy only)

Output files (written to --output_dir):
  how2sign_face_train_sequences.npy
  how2sign_face_train_labels.npy
  how2sign_face_train_meta.json
  how2sign_face_val_*
  how2sign_face_test_*
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import load_sequence_face, save_split


def process_how2sign_face(body_dir, face_dir, split, feature_type,
                          max_files=None):
    """
    Process all JSON files in body_dir for one split.
    Files without a matching face JSON are skipped with a warning.

    Parameters
    ----------
    body_dir     : str | Path
    face_dir     : str | Path
    split        : str   "train" | "val" | "test"
    feature_type : str   "optical_flow" | "raw_xy"
    max_files    : int | None

    Returns
    -------
    seqs  : list of float32 ndarrays
    labs  : list of int8 ndarrays  (all ones)
    metas : list of dicts
    """
    files = sorted(Path(body_dir).glob("*.json"))
    if max_files:
        files = files[:max_files]

    seqs, labs, metas = [], [], []

    for i, body_path in enumerate(files, 1):
        face_path = Path(face_dir) / body_path.name
        if not face_path.exists():
            print(f"    [WARN] No face JSON for {body_path.name}, skipping")
            continue

        print(f"  [{i}/{len(files)}] How2Sign-face {split}: {body_path.name}",
              flush=True)
        try:
            features, _, _, _ = load_sequence_face(
                str(body_path), str(face_path), feature_type)
            if features is None:
                continue
            T = len(features)
            labels = np.ones(T, dtype=np.int8)
            seqs.append(features)
            labs.append(labels)
            metas.append({
                "source":   "how2sign",
                "file":     body_path.name,
                "split":    split,
                "n_frames": T,
                "features": "body+hand+face",
            })
        except Exception as e:
            print(f"    [WARN] {body_path.name}: {e}")

    print(f"  -> How2Sign-face {split}: {len(seqs)} sequences")
    return seqs, labs, metas


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess How2Sign body+face keypoints into feature arrays.")
    parser.add_argument("--train",        required=True,
                        help="Body keypoint JSON directory — train split")
    parser.add_argument("--val",          required=True,
                        help="Body keypoint JSON directory — val split")
    parser.add_argument("--test",         required=True,
                        help="Body keypoint JSON directory — test split")
    parser.add_argument("--face_train",   required=True,
                        help="Face keypoint JSON directory — train split")
    parser.add_argument("--face_val",     required=True,
                        help="Face keypoint JSON directory — val split")
    parser.add_argument("--face_test",    required=True,
                        help="Face keypoint JSON directory — test split")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write intermediate .npy / .json files")
    parser.add_argument("--feature_type", choices=["optical_flow", "raw_xy"],
                        default="optical_flow")
    parser.add_argument("--test_mode",    action="store_true",
                        help="Process only the first 10 files per split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    max_files = 10 if args.test_mode else None

    print(f"Feature type: {args.feature_type}  (body+hand+face, dim=167/334)")
    print(f"{'[TEST MODE]' if args.test_mode else '[FULL MODE]'}\n")

    for split, bdir, fdir in [
        ("train", args.train, args.face_train),
        ("val",   args.val,   args.face_val),
        ("test",  args.test,  args.face_test),
    ]:
        print(f"=== How2Sign-face {split} ===")
        seqs, labs, metas = process_how2sign_face(
            bdir, fdir, split, args.feature_type, max_files)
        save_split(args.output_dir, f"how2sign_face_{split}", seqs, labs, metas)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
