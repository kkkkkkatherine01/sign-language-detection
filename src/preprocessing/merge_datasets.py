import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import save_split

SPLITS = ["train", "val", "test"]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_intermediate(input_dir, dataset, split):
    """
    Load one dataset / split's sequences, labels, and metadata.

    Returns
    -------
    seqs  : list of float32 ndarrays
    labs  : list of int8 ndarrays
    metas : list of dicts
    """
    prefix = os.path.join(input_dir, f"{dataset}_{split}")

    seq_path  = f"{prefix}_sequences.npy"
    lab_path  = f"{prefix}_labels.npy"
    meta_path = f"{prefix}_meta.json"

    for p in (seq_path, lab_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing intermediate file: {p}")

    seqs  = list(np.load(seq_path,  allow_pickle=True))
    labs  = list(np.load(lab_path,  allow_pickle=True))
    with open(meta_path) as f:
        metas = json.load(f)

    return seqs, labs, metas


# ── Merging ───────────────────────────────────────────────────────────────────

def merge_split(input_dir, datasets, split, shuffle=False, seed=42):
    """
    Load and concatenate all datasets for one split.

    Parameters
    ----------
    input_dir : str
    datasets  : list of str   e.g. ["how2sign", "dgs", "ted"]
    split     : str           "train" | "val" | "test"
    shuffle   : bool
    seed      : int

    Returns
    -------
    seqs, labs, metas (concatenated lists)
    """
    all_seqs, all_labs, all_metas = [], [], []

    for dataset in datasets:
        seqs, labs, metas = load_intermediate(input_dir, dataset, split)
        print(f"    {dataset:12s}: {len(seqs):5d} sequences")
        all_seqs  += seqs
        all_labs  += labs
        all_metas += metas

    if shuffle:
        rng     = random.Random(seed)
        indices = list(range(len(all_seqs)))
        rng.shuffle(indices)
        all_seqs  = [all_seqs[i]  for i in indices]
        all_labs  = [all_labs[i]  for i in indices]
        all_metas = [all_metas[i] for i in indices]
        print(f"    → shuffled (seed={seed})")

    return all_seqs, all_labs, all_metas


# ── Split-map copy ────────────────────────────────────────────────────────────

def copy_split_maps(input_dir, output_dir):
    """Copy all *_split_map.json files from input_dir to output_dir."""
    copied = []
    for p in sorted(Path(input_dir).glob("*_split_map.json")):
        dest = os.path.join(output_dir, p.name)
        shutil.copy2(p, dest)
        copied.append(p.name)
    if copied:
        print(f"  Split maps copied: {', '.join(copied)}")
    else:
        print("  [INFO] No split_map.json files found in input_dir.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge per-dataset intermediate files into final format.")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing intermediate .npy / .json files")
    parser.add_argument("--datasets",   required=True, nargs="+",
                        help="Datasets to merge, in order  e.g. how2sign dgs ted")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the final merged files")
    parser.add_argument("--shuffle",    action="store_true",
                        help="Shuffle sequence order within each split")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed used when --shuffle is set")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Datasets:   {' + '.join(args.datasets)}")
    print(f"Shuffle:    {args.shuffle}" + (f"  (seed={args.seed})" if args.shuffle else ""))
    print()

    for split in SPLITS:
        print(f"=== {split} ===")
        seqs, labs, metas = merge_split(
            args.input_dir, args.datasets, split,
            shuffle=args.shuffle, seed=args.seed,
        )
        save_split(args.output_dir, split, seqs, labs, metas)
        print()

    print("=== Split maps ===")
    copy_split_maps(args.input_dir, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
