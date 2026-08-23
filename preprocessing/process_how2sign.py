"""
process_how2sign.py

Preprocess How2Sign keypoint JSON files into per-frame features and labels.

Output files (written to --output_dir):
  how2sign_train_sequences.npy
  how2sign_train_labels.npy
  how2sign_train_meta.json
  how2sign_val_*
  how2sign_test_*
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import load_sequence, save_split


def process_how2sign(keypoint_dir, split, feature_type):
    """Process all JSON files in keypoint_dir for one split."""
    files = sorted(Path(keypoint_dir).glob("*.json"))

    seqs, labs, metas = [], [], []

    for i, path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] How2Sign {split}: {path.name}", flush=True)
        try:
            features, _, _, _ = load_sequence(str(path), feature_type)
            if features is None:
                continue
            T = len(features)
            labels = np.ones(T, dtype=np.int8)
            seqs.append(features)
            labs.append(labels)
            metas.append({
                "source":  "how2sign",
                "file":    path.name,
                "split":   split,
                "n_frames": T,
            })
        except Exception as e:
            print(f"    [WARN] {path.name}: {e}")

    print(f"  -> How2Sign {split}: {len(seqs)} sequences")
    return seqs, labs, metas


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess How2Sign keypoints into feature arrays.")
    parser.add_argument("--train",        required=True,
                        help="Directory of train split keypoint JSONs")
    parser.add_argument("--val",          required=True,
                        help="Directory of val split keypoint JSONs")
    parser.add_argument("--test",         required=True,
                        help="Directory of test split keypoint JSONs")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write intermediate .npy / .json files")
    parser.add_argument("--feature_type", choices=["optical_flow", "raw_xy"],
                        default="optical_flow")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Feature type: {args.feature_type}\n")

    for split, kdir in [("train", args.train),
                        ("val",   args.val),
                        ("test",  args.test)]:
        print(f"=== How2Sign {split} ===")
        seqs, labs, metas = process_how2sign(kdir, split, args.feature_type)
        save_split(args.output_dir, f"how2sign_{split}", seqs, labs, metas)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
