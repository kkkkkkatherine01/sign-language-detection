"""
process_ted.py

Preprocess TED Talks keypoint JSON files into per-frame features and labels.

All TED frames are labelled signing=0 (non-signing background class).

Outlier filtering uses a combined criterion (TED-specific):
  - motion magnitude > MOTION_THRESHOLD  (p95 of the distribution)
  AND
  - shoulder jump     > SHOULDER_JUMP_THRESHOLD  (p99 of shoulder displacement)

This filters camera cuts and pose estimation failures while preserving
genuine large-motion gestures, which would be flagged by motion alone.

Output files (written to --output_dir):
  ted_split_map.json
  ted_train_sequences.npy  /  ted_train_labels.npy  /  ted_train_meta.json
  ted_val_*
  ted_test_*
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import (
    compute_frame_motion,
    compute_shoulder_jump,
    frame_to_coords,
    load_sequence,
    make_video_split,
    save_split,
)

# ── TED-specific outlier thresholds ──────────────────────────────────────────
# Derived from analyze_ted_outliers.py and analyze_shoulder_jumps.py.
# A frame is an outlier only if BOTH conditions are met:
#   1. mean keypoint motion > p95  (unusual motion magnitude)
#   2. shoulder jump > sj_p99      (camera cut or pose estimation failure)
# Frames with high motion but no shoulder jump are genuine large gestures
# and should NOT be removed.

MOTION_THRESHOLD         = 32.29   # p95 of TED motion magnitude distribution
SHOULDER_JUMP_THRESHOLD  = 0.0562  # p99 of TED shoulder jump distribution


# ── TED-specific sequence loader ─────────────────────────────────────────────

def load_ted_sequence(json_path, feature_type):
    """
    Load a TED keypoint JSON and apply combined outlier filtering.

    Outlier condition: motion > MOTION_THRESHOLD AND shoulder_jump > SHOULDER_JUMP_THRESHOLD

    Returns
    -------
    features   : float32 ndarray or None
    T_valid    : int
    fps        : float
    n_removed  : int
    n_original : int
    """
    features, T_valid, fps, frames = load_sequence(json_path, feature_type)
    if features is None or len(frames) < 2:
        return None, None, fps, 0, len(frames) if frames else 0

    n_original = len(frames)

    # Rebuild coords_seq from raw frames (same forward-fill as load_sequence)
    raw = [frame_to_coords(f) for f in frames]
    first_valid = next((c for c in raw if c is not None), None)
    if first_valid is None:
        return None, None, fps, 0, n_original
    last = first_valid
    coords_seq = []
    for c in raw:
        if c is None:
            coords_seq.append(last)
        else:
            last = c
            coords_seq.append(c)

    motion = compute_frame_motion(coords_seq, fps)          # (T-1,)
    shoulder = compute_shoulder_jump(frames)                 # (T-1,)

    # Outlier: motion AND shoulder both exceed thresholds
    # Index 0 of features corresponds to transition 0→1, same as motion[0].
    outlier_mask = (
        (motion        > MOTION_THRESHOLD) &
        (shoulder      > SHOULDER_JUMP_THRESHOLD)
    )                                                        # (T-1,) bool

    n_removed = int(outlier_mask.sum())
    if n_removed == 0:
        return features, T_valid, fps, 0, n_original

    if feature_type == "optical_flow":
        # outlier_mask[i] marks transition i→i+1; features[i] is that transition.
        features = features[~outlier_mask]
    else:
        # raw_xy: features[i] is frame i. Remove the "arrived-at" frame i+1.
        frame_keep = np.ones(len(features), dtype=bool)
        frame_keep[1:][outlier_mask] = False
        features = features[frame_keep]

    if len(features) == 0:
        return None, None, fps, n_removed, n_original

    return features, len(features), fps, n_removed, n_original


# ── Processing ────────────────────────────────────────────────────────────────

def process_ted(keypoint_dir, feature_type, split_map, max_files=None):
    files = sorted(Path(keypoint_dir).glob("*.json"))
    if max_files:
        files = files[:max_files]

    seqs_s  = {"train": [], "val": [], "test": []}
    labs_s  = {"train": [], "val": [], "test": []}
    metas_s = {"train": [], "val": [], "test": []}
    total_original = total_removed = 0

    for i, path in enumerate(files, 1):
        video_id = path.stem
        split    = split_map.get(video_id, "train")
        print(f"  [{i}/{len(files)}] TED {split}: {path.name}", flush=True)
        try:
            features, _, _, n_removed, n_original = load_ted_sequence(
                str(path), feature_type)
            if features is None:
                continue

            total_original += n_original
            total_removed  += n_removed
            T = len(features)
            labels = np.zeros(T, dtype=np.int8)

            seqs_s[split].append(features)
            labs_s[split].append(labels)
            metas_s[split].append({
                "source":             "ted",
                "file":               path.name,
                "split":              split,
                "n_frames":           T,
                "n_outliers_removed": n_removed,
            })
        except Exception as e:
            print(f"    [WARN] {path.name}: {e}")

    for s in ["train", "val", "test"]:
        print(f"  -> TED {s}: {len(seqs_s[s])} sequences")

    if total_original > 0:
        pct = 100.0 * total_removed / total_original
        print(f"\n  Outlier filtering (motion>p95 AND shoulder_jump>p99): "
              f"removed {total_removed:,} / {total_original:,} frames ({pct:.2f}%)")

    return seqs_s, labs_s, metas_s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TED Talks keypoints into feature arrays.")
    parser.add_argument("--ted_dir",      required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--feature_type", choices=["optical_flow", "raw_xy"],
                        default="optical_flow")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--test_mode",    action="store_true",
                        help="Process only the first 10 files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    max_files = 10 if args.test_mode else None

    print(f"Feature type: {args.feature_type}")
    print(f"Outlier filter: motion > {MOTION_THRESHOLD} (p95) "
          f"AND shoulder_jump > {SHOULDER_JUMP_THRESHOLD} (p99)")
    print(f"{'[TEST MODE]' if args.test_mode else '[FULL MODE]'}\n")

    all_ids   = [p.stem for p in Path(args.ted_dir).glob("*.json")]
    split_map = make_video_split(all_ids, seed=args.seed)
    split_map_path = os.path.join(args.output_dir, "ted_split_map.json")
    with open(split_map_path, "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"Split map saved → {split_map_path}  ({len(all_ids)} videos)\n")

    seqs_s, labs_s, metas_s = process_ted(
        args.ted_dir, args.feature_type, split_map, max_files)

    print("\n=== Saving ===")
    for s in ["train", "val", "test"]:
        save_split(args.output_dir, f"ted_{s}",
                   seqs_s[s], labs_s[s], metas_s[s])
    print("\nDone.")


if __name__ == "__main__":
    main()
