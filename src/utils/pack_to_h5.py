import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

SPLITS = ["train", "val", "test"]


def pack_split(h5_file, input_dir, split):
    seqs = np.load(os.path.join(input_dir, f"{split}_sequences.npy"), allow_pickle=True)
    labs = np.load(os.path.join(input_dir, f"{split}_labels.npy"),    allow_pickle=True)

    lengths        = np.array([len(s) for s in seqs], dtype=np.int64)
    sequences_flat = np.concatenate(list(seqs), axis=0).astype(np.float32)
    labels_flat    = np.concatenate(list(labs), axis=0).astype(np.int8)

    grp = h5_file.create_group(split)
    grp.create_dataset("sequences", data=sequences_flat,
                       compression="gzip", compression_opts=4)
    grp.create_dataset("lengths",   data=lengths)
    grp.create_dataset("labels",    data=labels_flat,
                       compression="gzip", compression_opts=4)

    meta_path = os.path.join(input_dir, f"{split}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            grp.attrs["meta"] = f.read()
    else:
        grp.attrs["meta"] = "[]"

    total    = int(lengths.sum())
    pos      = int((labels_flat > 0).sum())
    feat_dim = sequences_flat.shape[1]
    print(f"  {split:5s}: {len(seqs):5d} seqs | {total:>12,} frames | "
          f"signing={100 * pos / max(total, 1):.1f}% | feat_dim={feat_dim}")
    return feat_dim


def main():
    parser = argparse.ArgumentParser(
        description="Pack merged .npy output into a single HDF5 file.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory with {split}_sequences.npy / _labels.npy / _meta.json")
    parser.add_argument("--output",    required=True,
                        help="Output .h5 file path  (e.g. dataset_optical_flow_bh.h5)")
    args = parser.parse_args()

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output}\n")

    os.makedirs(Path(args.output).parent, exist_ok=True)

    feat_dims = set()
    with h5py.File(args.output, "w") as f:
        for split in SPLITS:
            seq_path = os.path.join(args.input_dir, f"{split}_sequences.npy")
            if not os.path.exists(seq_path):
                print(f"  [{split}] not found, skipping")
                continue
            feat_dims.add(pack_split(f, args.input_dir, split))

        if feat_dims:
            f.attrs["feat_dim"] = min(feat_dims)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone → {args.output}  ({size_mb:.0f} MB, 1 file)")


if __name__ == "__main__":
    main()
