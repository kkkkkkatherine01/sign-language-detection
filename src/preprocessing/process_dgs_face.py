"""
process_dgs_face.py

Same as process_dgs.py, but with body+hand+face keypoints

Output files (written to --output_dir):
  dgs_face_split_map.json
  dgs_face_train_sequences.npy  /  dgs_face_train_labels.npy  /  dgs_face_train_meta.json
  dgs_face_val_*
  dgs_face_test_*
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
from preprocessing_utils import (
    load_sequence_face,
    make_video_split,
    save_split,
)

DGS_DOWNSAMPLE_FACTOR = 2   # 50 fps → 25 fps


def parse_eaf_sentence(eaf_path, signer):
    """Same as process_dgs.py."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        tsid = ts.get("TIME_SLOT_ID")
        val  = ts.get("TIME_VALUE")
        if val is not None:
            time_slots[tsid] = int(val)

    intervals   = []
    target_tier = f"Deutsche_Übersetzung_{signer}"
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
    """Same as process_dgs.py."""
    labels = np.zeros(num_frames, dtype=np.int8)
    for start_ms, end_ms in intervals:
        s = max(0,              math.ceil(start_ms / 1000.0 * fps))
        e = min(num_frames - 1, int(end_ms         / 1000.0 * fps))
        labels[s:e + 1] = 1
    return labels


def process_dgs_signer_face(body_dir, face_dir, eaf_dir, feature_type,
                             signer, split_map):
    """Process all JSON files for one signer (body+face)."""
    files = sorted(Path(body_dir).glob("*.json"))

    seqs_s  = {"train": [], "val": [], "test": []}
    labs_s  = {"train": [], "val": [], "test": []}
    metas_s = {"train": [], "val": [], "test": []}

    for i, body_path in enumerate(files, 1):
        face_path  = Path(face_dir) / body_path.name
        numeric_id = body_path.stem.split("_")[0]
        eaf_path   = Path(eaf_dir) / f"{numeric_id}.eaf"
        split      = split_map.get(numeric_id, "train")

        if not face_path.exists():
            print(f"    [WARN] No face JSON for {body_path.name}, skipping")
            continue
        if not eaf_path.exists():
            print(f"    [WARN] No EAF for {numeric_id}, skipping")
            continue

        print(f"  [{i}/{len(files)}] DGS-face {signer} {split}: {body_path.name}",
              flush=True)
        try:
            features, T_valid, fps, _ = load_sequence_face(
                str(body_path), str(face_path), feature_type,
                downsample_factor=DGS_DOWNSAMPLE_FACTOR,
            )
            if features is None:
                continue

            intervals   = parse_eaf_sentence(str(eaf_path), signer)
            labels_full = get_frame_labels(T_valid, fps, intervals)

            if feature_type == "optical_flow":
                labels = labels_full[1:]
            else:
                labels = labels_full

            T = len(features)
            assert len(labels) == T, (
                f"Label/feature length mismatch: {len(labels)} vs {T} "
                f"in {body_path.name}"
            )

            seqs_s[split].append(features)
            labs_s[split].append(labels)
            metas_s[split].append({
                "source":        "dgs",
                "signer":        signer,
                "file":          body_path.name,
                "split":         split,
                "n_frames":      T,
                "fps":           fps,
                "label_tier":    "sentence",
                "features":      "body+hand+face",
                "signing_ratio": float(labels.mean()),
            })
        except Exception as e:
            print(f"    [WARN] {body_path.name}: {e}")

    for s in ["train", "val", "test"]:
        n   = len(seqs_s[s])
        tot = sum(len(l) for l in labs_s[s])
        pos = sum(int(l.sum()) for l in labs_s[s])
        pct = 100 * pos / tot if tot else 0
        print(f"  -> DGS-face {signer} {s}: {n} seqs | {tot:,} frames | "
              f"signing={pos:,} ({pct:.1f}%)")

    return seqs_s, labs_s, metas_s


def process_dgs_face(body_dir_a, face_dir_a, body_dir_b, face_dir_b,
                     eaf_dir, feature_type, split_map):
    """Process signer A and (optionally) signer B, merging per split."""
    seqs_s  = {"train": [], "val": [], "test": []}
    labs_s  = {"train": [], "val": [], "test": []}
    metas_s = {"train": [], "val": [], "test": []}

    print("  --- Signer A ---")
    Xa, ya, ma = process_dgs_signer_face(
        body_dir_a, face_dir_a, eaf_dir, feature_type, "A", split_map)
    for s in ["train", "val", "test"]:
        seqs_s[s] += Xa[s]
        labs_s[s] += ya[s]
        metas_s[s] += ma[s]

    if body_dir_b is not None:
        print("  --- Signer B ---")
        Xb, yb, mb = process_dgs_signer_face(
            body_dir_b, face_dir_b, eaf_dir, feature_type, "B", split_map)
        for s in ["train", "val", "test"]:
            seqs_s[s] += Xb[s]
            labs_s[s] += yb[s]
            metas_s[s] += mb[s]

    return seqs_s, labs_s, metas_s


def build_dgs_split_map(body_dir_a, body_dir_b, eaf_dir, seed=42):
    """Same as process_dgs.py."""
    eaf_dir_path = Path(eaf_dir)

    def has_eaf(p):
        return (eaf_dir_path / f"{p.stem.split('_')[0]}.eaf").exists()

    ids_a = {p.stem.split("_")[0]
             for p in Path(body_dir_a).glob("*.json") if has_eaf(p)}
    ids_b = {p.stem.split("_")[0]
             for p in Path(body_dir_b).glob("*.json") if has_eaf(p)} \
            if body_dir_b is not None else set()

    all_ids = sorted(ids_a | ids_b)
    print(f"DGS sessions with EAF: A={len(ids_a)} B={len(ids_b)} "
          f"union={len(all_ids)}")

    return make_video_split(all_ids, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DGS Corpus body+face keypoints into feature arrays.")
    parser.add_argument("--dgs_dir_a",      required=True,
                        help="Body keypoint JSON directory — signer A")
    parser.add_argument("--dgs_face_dir_a", required=True,
                        help="Face keypoint JSON directory — signer A")
    parser.add_argument("--dgs_dir_b",      default=None,
                        help="Body keypoint JSON directory — signer B (optional)")
    parser.add_argument("--dgs_face_dir_b", default=None,
                        help="Face keypoint JSON directory — signer B (optional)")
    parser.add_argument("--eaf_dir",        required=True,
                        help="Directory containing .eaf annotation files")
    parser.add_argument("--output_dir",     required=True,
                        help="Where to write intermediate .npy / .json files")
    parser.add_argument("--feature_type",   choices=["optical_flow", "raw_xy"],
                        default="optical_flow")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    if args.dgs_dir_b is not None and args.dgs_face_dir_b is None:
        parser.error("--dgs_face_dir_b is required when --dgs_dir_b is provided")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Feature type:      {args.feature_type}  (body+hand+face, dim=167/334)")
    print(f"Downsample factor: {DGS_DOWNSAMPLE_FACTOR}  (50 fps → 25 fps)\n")

    split_map = build_dgs_split_map(
        args.dgs_dir_a, args.dgs_dir_b, args.eaf_dir, seed=args.seed)
    split_map_path = os.path.join(args.output_dir, "dgs_face_split_map.json")
    with open(split_map_path, "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"Split map saved → {split_map_path}\n")

    seqs_s, labs_s, metas_s = process_dgs_face(
        args.dgs_dir_a, args.dgs_face_dir_a,
        args.dgs_dir_b, args.dgs_face_dir_b,
        args.eaf_dir, args.feature_type, split_map,
    )

    print("\n=== Saving ===")
    for s in ["train", "val", "test"]:
        save_split(args.output_dir, f"dgs_face_{s}",
                   seqs_s[s], labs_s[s], metas_s[s])

    print("\nDone.")


if __name__ == "__main__":
    main()
