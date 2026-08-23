"""
process_sslc_face.py

Same as process_sslc.py, but with body+hand+face keypoints

Output files (written to --output_dir):
  sslc_face_sequences.npy
  sslc_face_labels.npy
  sslc_face_meta.json
"""

import argparse
import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing_utils import (
    get_shoulder_width,
    frame_to_coords_face,
    compute_optical_flow,
    compute_raw_xy,
    save_split,
)


def parse_eaf_sentence(eaf_path, signer):
    """Same as process_sslc.py."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots = {}
    for ts in root.findall('.//TIME_SLOT'):
        tsid = ts.get('TIME_SLOT_ID')
        val  = ts.get('TIME_VALUE')
        if val is not None:
            time_slots[tsid] = int(val)

    intervals = []
    tier_name = f"Översättning {signer}"
    for tier in root.findall('.//TIER'):
        if tier.get('TIER_ID', '') != tier_name:
            continue
        for ann in tier.findall('.//ALIGNABLE_ANNOTATION'):
            t1 = ann.get('TIME_SLOT_REF1')
            t2 = ann.get('TIME_SLOT_REF2')
            if t1 in time_slots and t2 in time_slots:
                intervals.append((time_slots[t1], time_slots[t2]))

    return intervals


def get_frame_labels(num_frames, fps, intervals):
    """Same as process_sslc.py."""
    labels = np.zeros(num_frames, dtype=np.int8)
    for start_ms, end_ms in intervals:
        s = max(0,              math.ceil(start_ms / 1000.0 * fps))
        e = min(num_frames - 1, int(end_ms         / 1000.0 * fps))
        labels[s:e + 1] = 1
    return labels


def load_sequence_sslc_face(body_json_path, face_json_path, feature_type):
    """
    Load SSLC body+face keypoint JSONs and return features + metadata.

    Shoulder width (from body) is used to normalise both body and face.
    Frames before the first valid pose detection are trimmed (disclaimer
    footage); subsequent invalid frames are forward-filled.

    Returns (features, T_valid, fps, total_frames, first_valid_idx) — features
    is None if no valid pose was ever detected.
    """
    with open(body_json_path) as f:
        body_data = json.load(f)
    with open(face_json_path) as f:
        face_data = json.load(f)

    body_frames  = body_data['frames']
    face_frames  = face_data['frames']
    fps          = float(body_data['fps'])
    total_frames = int(body_data['total_frames'])
    n_frames     = min(len(body_frames), len(face_frames))

    raw = []
    for i in range(n_frames):
        sw = get_shoulder_width(body_frames[i]['pose'])
        raw.append(frame_to_coords_face(body_frames[i], face_frames[i], sw))

    first_valid_idx = next(
        (i for i, c in enumerate(raw) if c is not None), None)
    if first_valid_idx is None:
        return None, None, fps, total_frames, 0

    raw = raw[first_valid_idx:]

    last_valid = next(c for c in raw if c is not None)
    coords_seq = []
    for coords in raw:
        if coords is None:
            coords_seq.append(last_valid)
        else:
            last_valid = coords
            coords_seq.append(coords)

    if len(coords_seq) < 2:
        return None, None, fps, total_frames, first_valid_idx

    T_valid = len(coords_seq)

    if feature_type == 'optical_flow':
        features = compute_optical_flow(coords_seq, fps)
    else:
        features = compute_raw_xy(coords_seq)

    return features, T_valid, fps, total_frames, first_valid_idx


def process_sslc_face(pairs_file, keypoint_dir, face_keypoint_dir,
                      eaf_dir, feature_type):
    rows = []
    with open(pairs_file, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['video_file'].strip().endswith('.mp4'):
                rows.append(row)

    seqs  = []
    labs  = []
    metas = []

    skipped_missing = 0
    skipped_empty   = 0

    for i, row in enumerate(rows, 1):
        vf       = row['video_file'].strip()
        signer   = row['signer'].strip()
        eaf_name = row['eaf_file'].strip()
        video_id = Path(vf).stem

        body_json = Path(keypoint_dir)      / f"{video_id}.json"
        face_json = Path(face_keypoint_dir) / f"{video_id}.json"
        eaf_path  = Path(eaf_dir) / eaf_name

        if not body_json.exists():
            print(f"  [WARN] Body JSON not found: {body_json}")
            skipped_missing += 1
            continue
        if not face_json.exists():
            print(f"  [WARN] Face JSON not found: {face_json}")
            skipped_missing += 1
            continue
        if not eaf_path.exists():
            print(f"  [WARN] EAF not found: {eaf_path}")
            skipped_missing += 1
            continue

        print(f"  [{i}/{len(rows)}] {video_id} {signer}", flush=True)

        try:
            features, T_valid, fps, total_frames, first_valid_idx = \
                load_sequence_sslc_face(
                    str(body_json), str(face_json), feature_type)

            if features is None:
                skipped_empty += 1
                continue

            intervals   = parse_eaf_sentence(str(eaf_path), signer)
            labels_full = get_frame_labels(total_frames, fps, intervals)
            labels_trim = labels_full[first_valid_idx:]

            if feature_type == 'optical_flow':
                labels = labels_trim[1:len(features) + 1]
            else:
                labels = labels_trim[:len(features)]

            T = len(features)
            assert len(labels) == T, (
                f"Label/feature mismatch: {len(labels)} vs {T} "
                f"in {video_id}")

            seqs.append(features)
            labs.append(labels)
            metas.append({
                'source':          'sslc',
                'signer':          signer,
                'video_id':        video_id,
                'eaf':             eaf_name,
                'n_frames':        T,
                'fps':             fps,
                'label_tier':      'sentence',
                'features':        'body+hand+face',
                'signing_ratio':   float(labels.mean()),
                'first_valid_idx': first_valid_idx,
            })

        except Exception as e:
            print(f"  [WARN] {video_id}: {e}")
            skipped_empty += 1

    print(f"\nProcessed:         {len(seqs)}")
    print(f"Skipped (missing): {skipped_missing}")
    print(f"Skipped (empty):   {skipped_empty}")

    return seqs, labs, metas


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SSLC body+hand+face keypoints into feature arrays.")
    parser.add_argument('--pairs_file',        required=True,
                        help="Path to sslc_available_pairs.txt")
    parser.add_argument('--keypoint_dir',      required=True,
                        help="Directory containing body+hand JSON files")
    parser.add_argument('--face_keypoint_dir', required=True,
                        help="Directory containing face JSON files")
    parser.add_argument('--eaf_dir',           required=True,
                        help="Directory containing .eaf annotation files")
    parser.add_argument('--output_dir',        required=True,
                        help="Where to write output .npy / .json files")
    parser.add_argument('--feature_type',      choices=['optical_flow', 'raw_xy'],
                        default='optical_flow')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Feature type: {args.feature_type}  (body+hand+face, dim=167/334)")
    print("FPS:          25.0 (no downsampling)\n")

    seqs, labs, metas = process_sslc_face(
        args.pairs_file, args.keypoint_dir, args.face_keypoint_dir,
        args.eaf_dir, args.feature_type)

    print('\n=== Saving ===')
    save_split(args.output_dir, 'sslc_face', seqs, labs, metas)
    print('\nDone.')


if __name__ == '__main__':
    main()
