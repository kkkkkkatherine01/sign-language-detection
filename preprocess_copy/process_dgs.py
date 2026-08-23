"""
process_dgs.py

Preprocess DGS Corpus keypoint JSON files into per-frame features and labels.

Labels come from sentence-level EAF annotations (Deutsche_Übersetzung tier).
Videos without a matching EAF file are excluded from both the split map and
processing.

DGS is recorded at 50 fps and is downsampled to 25 fps (every other frame).
The returned fps is updated accordingly so optical flow magnitudes and
label-to-frame alignment remain correct.

Output files (written to --output_dir):
  dgs_split_map.json
  dgs_train_sequences.npy  /  dgs_train_labels.npy  /  dgs_train_meta.json
  dgs_val_*
  dgs_test_*
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import load_sequence, make_video_split, save_split

DGS_DOWNSAMPLE_FACTOR = 2   # 50 fps → 25 fps


# ── EAF parsing (DGS-specific) ────────────────────────────────────────────────

def parse_eaf_sentence(eaf_path, signer):
    """
    Extract signing intervals from the Deutsche_Übersetzung_{signer} tier.

    Returns
    -------
    list of (start_ms, end_ms) tuples
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        tsid = ts.get("TIME_SLOT_ID")
        val  = ts.get("TIME_VALUE")
        if val is not None:
            time_slots[tsid] = int(val)

    intervals    = []
    target_tier  = f"Deutsche_Übersetzung_{signer}"
    for tier in root.findall(".//TIER"):
        if tier.get("TIER_ID", "") != target_tier:
            continue
        for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
            t1 = ann.get("TIME_SLOT_REF1")
            t2 = ann.get("TIME_SLOT_REF2")
            if t1 in time_slots and t2 in time_slots:
                intervals.append((time_slots[t1], time_slots[t2]))

    return intervals


def get_frame_labels(num_frames, fps, intervals):
    """
    Build a per-frame int8 label array from ms-level annotation intervals.

    Uses ceil for the start frame (first frame fully inside the interval)
    and floor for the end frame.
    """
    labels = np.zeros(num_frames, dtype=np.int8)
    for start_ms, end_ms in intervals:
        s = max(0,              math.ceil(start_ms / 1000.0 * fps))
        e = min(num_frames - 1, int(end_ms   / 1000.0 * fps))
        labels[s:e + 1] = 1
    return labels


# ── Per-signer processing ─────────────────────────────────────────────────────

def process_dgs_signer(keypoint_dir, eaf_dir, feature_type, signer,
                       split_map, max_files=None):
    """
    Process all JSON files for one signer.

    Parameters
    ----------
    keypoint_dir : str | Path
    eaf_dir      : str | Path
    feature_type : "optical_flow" | "raw_xy"
    signer       : "A" | "B"
    split_map    : dict  numeric_id → "train" | "val" | "test"
    max_files    : int | None

    Returns
    -------
    seqs_s, labs_s, metas_s : dicts keyed by split name
    """
    files = sorted(Path(keypoint_dir).glob("*.json"))
    if max_files:
        files = files[:max_files]

    seqs_s  = {"train": [], "val": [], "test": []}
    labs_s  = {"train": [], "val": [], "test": []}
    metas_s = {"train": [], "val": [], "test": []}

    for i, path in enumerate(files, 1):
        numeric_id = path.stem.split("_")[0]
        eaf_path   = Path(eaf_dir) / f"{numeric_id}.eaf"
        split      = split_map.get(numeric_id, "train")

        # Videos excluded at split-map build time should not appear here,
        # but guard anyway in case the directory has extra files.
        if not eaf_path.exists():
            print(f"    [WARN] No EAF for {numeric_id}, skipping")
            continue

        print(f"  [{i}/{len(files)}] DGS {signer} {split}: {path.name}",
              flush=True)
        try:
            features, T_valid, fps, _ = load_sequence(
                str(path), feature_type,
                downsample_factor=DGS_DOWNSAMPLE_FACTOR,
            )
            if features is None:
                continue

            intervals   = parse_eaf_sentence(str(eaf_path), signer)
            # fps is already the downsampled fps (25), so frame indices are correct.
            labels_full = get_frame_labels(T_valid, fps, intervals)

            if feature_type == "optical_flow":
                labels = labels_full[1:]
            else:
                labels = labels_full

            T = len(features)
            assert len(labels) == T, (
                f"Label/feature length mismatch: {len(labels)} vs {T} "
                f"in {path.name}"
            )

            seqs_s[split].append(features)
            labs_s[split].append(labels)
            metas_s[split].append({
                "source":        "dgs",
                "signer":        signer,
                "file":          path.name,
                "split":         split,
                "n_frames":      T,
                "fps":           fps,
                "label_tier":    "sentence",
                "signing_ratio": float(labels.mean()),
            })
        except Exception as e:
            print(f"    [WARN] {path.name}: {e}")

    for s in ["train", "val", "test"]:
        n   = len(seqs_s[s])
        tot = sum(len(l) for l in labs_s[s])
        pos = sum(int(l.sum()) for l in labs_s[s])
        pct = 100 * pos / tot if tot else 0
        print(f"  -> DGS {signer} {s}: {n} seqs | {tot:,} frames | "
              f"signing={pos:,} ({pct:.1f}%)")

    return seqs_s, labs_s, metas_s


# ── Top-level DGS processing ──────────────────────────────────────────────────

def process_dgs(dgs_dir_a, dgs_dir_b, eaf_dir, feature_type,
                split_map, max_files=None):
    """
    Process signer A and (optionally) signer B, merging results per split.
    """
    seqs_s  = {"train": [], "val": [], "test": []}
    labs_s  = {"train": [], "val": [], "test": []}
    metas_s = {"train": [], "val": [], "test": []}

    print("  --- Signer A ---")
    Xa, ya, ma = process_dgs_signer(
        dgs_dir_a, eaf_dir, feature_type, "A", split_map, max_files)
    for s in ["train", "val", "test"]:
        seqs_s[s] += Xa[s]
        labs_s[s] += ya[s]
        metas_s[s] += ma[s]

    if dgs_dir_b is not None:
        print("  --- Signer B ---")
        Xb, yb, mb = process_dgs_signer(
            dgs_dir_b, eaf_dir, feature_type, "B", split_map, max_files)
        for s in ["train", "val", "test"]:
            seqs_s[s] += Xb[s]
            labs_s[s] += yb[s]
            metas_s[s] += mb[s]

    return seqs_s, labs_s, metas_s


# ── Split-map builder ─────────────────────────────────────────────────────────

def build_dgs_split_map(dgs_dir_a, dgs_dir_b, eaf_dir, seed=42):
    """
    Collect numeric IDs that have a matching EAF, then split train/val/test.

    EAF filtering happens here so that videos without annotations are never
    assigned to any split.
    """
    eaf_dir_path = Path(eaf_dir)

    def has_eaf(p):
        return (eaf_dir_path / f"{p.stem.split('_')[0]}.eaf").exists()

    ids_a = {p.stem.split("_")[0]
             for p in Path(dgs_dir_a).glob("*.json") if has_eaf(p)}
    ids_b = {p.stem.split("_")[0]
             for p in Path(dgs_dir_b).glob("*.json") if has_eaf(p)} \
            if dgs_dir_b is not None else set()

    all_ids = sorted(ids_a | ids_b)
    print(f"DGS sessions with EAF: A={len(ids_a)} B={len(ids_b)} "
          f"union={len(all_ids)}")

    return make_video_split(all_ids, seed=seed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DGS Corpus keypoints into feature arrays.")
    parser.add_argument("--dgs_dir_a",    required=True,
                        help="Keypoint JSON directory for signer A")
    parser.add_argument("--dgs_dir_b",    default=None,
                        help="Keypoint JSON directory for signer B (optional)")
    parser.add_argument("--eaf_dir",      required=True,
                        help="Directory containing .eaf annotation files")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to write intermediate .npy / .json files")
    parser.add_argument("--feature_type", choices=["optical_flow", "raw_xy"],
                        default="optical_flow")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--test_mode",    action="store_true",
                        help="Process only the first 10 files per signer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    max_files = 10 if args.test_mode else None

    print(f"Feature type:      {args.feature_type}")
    print(f"Downsample factor: {DGS_DOWNSAMPLE_FACTOR}  (50 fps → 25 fps)")
    print(f"{'[TEST MODE]' if args.test_mode else '[FULL MODE]'}\n")

    # Build and save split map (EAF-filtered).
    split_map = build_dgs_split_map(
        args.dgs_dir_a, args.dgs_dir_b, args.eaf_dir, seed=args.seed)
    split_map_path = os.path.join(args.output_dir, "dgs_split_map.json")
    with open(split_map_path, "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"Split map saved → {split_map_path}\n")

    # Process and save per split.
    seqs_s, labs_s, metas_s = process_dgs(
        args.dgs_dir_a, args.dgs_dir_b, args.eaf_dir,
        args.feature_type, split_map, max_files,
    )

    print("\n=== Saving ===")
    for s in ["train", "val", "test"]:
        save_split(args.output_dir, f"dgs_{s}",
                   seqs_s[s], labs_s[s], metas_s[s])

    print("\nDone.")


if __name__ == "__main__":
    main()
