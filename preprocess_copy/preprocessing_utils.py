"""
preprocessing_utils.py

Shared utilities for multi-dataset per-frame preprocessing.
Used by process_how2sign.py, process_dgs.py, process_ted.py,
      process_*_face.py, and merge_datasets.py.

Feature dims (body+hand only):
  optical_flow: 75   (per-keypoint L2 norm, 2D)
  raw_xy:      150   (75 × 2, xy only)

Feature dims (body+hand+face):
  optical_flow: 167  (75 body+hand + 92 face)
  raw_xy:       334  (167 × 2, xy only)
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np

# ── Skeleton constants ────────────────────────────────────────────────────────

POSE_N          = 33
HAND_N          = 21
FACE_N          = 92
BODY_HAND_N     = POSE_N + HAND_N * 2          # 75
TOTAL_N         = BODY_HAND_N + FACE_N          # 167
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12
LEG_INDICES     = set(range(23, 33))


# ── Shoulder normalisation ────────────────────────────────────────────────────

def get_shoulder_width(pose_landmarks):
    """Return Euclidean distance between shoulders, or None if landmarks are missing."""
    l = pose_landmarks[LEFT_SHOULDER]
    r = pose_landmarks[RIGHT_SHOULDER]
    lx, ly = l["x"], l["y"]
    rx, ry = r["x"], r["y"]
    if lx == 0 and ly == 0 and rx == 0 and ry == 0:
        return None
    dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    return dist if dist > 1e-6 else None


def normalize_landmarks(landmarks, sw):
    """Divide every (x, y) coordinate by shoulder width sw."""
    return [{"x": lm["x"] / sw, "y": lm["y"] / sw} for lm in landmarks]


# ── Frame → coordinate array ──────────────────────────────────────────────────

def frame_to_coords(frame):
    """
    Convert a single frame dict to a (BODY_HAND_N, 2) float32 array.

    Returns None if the shoulder width cannot be computed (missing landmarks).
    Leg landmarks (indices 23-32) are zeroed out.
    """
    pose       = frame["pose"]
    left_hand  = frame["left_hand"]
    right_hand = frame["right_hand"]

    sw = get_shoulder_width(pose)
    if sw is None:
        return None

    pose_n  = normalize_landmarks(pose,       sw)
    left_n  = normalize_landmarks(left_hand,  sw)
    right_n = normalize_landmarks(right_hand, sw)

    coords = []
    for idx, lm in enumerate(pose_n):
        if idx in LEG_INDICES:
            coords.extend([0.0, 0.0])
        else:
            coords.extend([lm["x"], lm["y"]])
    for lm in left_n + right_n:
        coords.extend([lm["x"], lm["y"]])

    return np.array(coords, dtype=np.float32).reshape(BODY_HAND_N, 2)


def frame_to_coords_face(frame_body, frame_face, sw):
    """
    Convert body + face frame dicts to a (TOTAL_N, 2) float32 array.

    sw must be pre-computed via get_shoulder_width(frame_body["pose"]).
    Returns None if sw is None (shoulder landmarks missing).
    Leg landmarks (body indices 23-32) are zeroed out.
    """
    if sw is None:
        return None

    pose_n  = normalize_landmarks(frame_body["pose"],       sw)
    left_n  = normalize_landmarks(frame_body["left_hand"],  sw)
    right_n = normalize_landmarks(frame_body["right_hand"], sw)
    face_n  = normalize_landmarks(frame_face["face"],       sw)

    coords = []
    for idx, lm in enumerate(pose_n):
        if idx in LEG_INDICES:
            coords.extend([0.0, 0.0])
        else:
            coords.extend([lm["x"], lm["y"]])
    for lm in left_n + right_n + face_n:
        coords.extend([lm["x"], lm["y"]])

    return np.array(coords, dtype=np.float32).reshape(TOTAL_N, 2)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_optical_flow(coords_seq, fps):
    """
    Per-keypoint L2 motion between consecutive frames, scaled by fps.

    Input:  list/array of T frames, each shape (BODY_HAND_N, 2)
    Output: float32 array of shape (T-1, BODY_HAND_N)
    """
    arr   = np.stack(coords_seq)
    delta = arr[1:] - arr[:-1]
    flow  = np.linalg.norm(delta, axis=2)
    return (flow * fps).astype(np.float32)


def compute_raw_xy(coords_seq):
    """
    Flatten (x, y) coordinates for every frame.

    Input:  list/array of T frames, each shape (N, 2)  — any N
    Output: float32 array of shape (T, N * 2)
    """
    arr = np.stack(coords_seq)
    return arr.reshape(len(coords_seq), -1).astype(np.float32)


def compute_frame_motion(coords_seq, fps):
    """
    Mean per-keypoint motion for each consecutive frame pair.

    Input:  list/array of T frames, each shape (N, 2)  — any N
    Output: float array of shape (T-1,)  — used for outlier filtering
    """
    arr   = np.stack(coords_seq)
    delta = arr[1:] - arr[:-1]
    flow  = np.linalg.norm(delta, axis=2)
    return (flow * fps).mean(axis=1)


def compute_frame_motion_bh(coords_seq, fps):
    """
    Mean per-keypoint motion using body+hand keypoints only.

    Used for outlier filtering in face mode: face landmarks (e.g. blinks,
    mouth movements) are noisier than body motion and should not drive the
    outlier threshold.

    Input:  list/array of T frames, each shape (TOTAL_N, 2)
    Output: float array of shape (T-1,)
    """
    arr   = np.stack(coords_seq)[:, :BODY_HAND_N, :]   # (T, 75, 2)
    delta = arr[1:] - arr[:-1]
    flow  = np.linalg.norm(delta, axis=2)
    return (flow * fps).mean(axis=1)


# ── Shoulder jump (raw coords) ────────────────────────────────────────────────

def compute_shoulder_jump(frames):
    """
    Compute per-frame shoulder midpoint displacement in raw (un-normalised)
    image coordinates.

    Used by TED outlier filtering to distinguish camera cuts / pose errors
    (large shoulder jump) from genuine large-motion gestures (no shoulder jump).

    Parameters
    ----------
    frames : list of frame dicts, each with a "pose" field

    Returns
    -------
    float32 array of shape (T-1,) — displacement between consecutive frames.
    Entry i corresponds to the transition from frame i to frame i+1.
    """
    midpoints = []
    last_mid  = None
    for frame in frames:
        pose = frame["pose"]
        l = pose[LEFT_SHOULDER]
        r = pose[RIGHT_SHOULDER]
        lx, ly = l["x"], l["y"]
        rx, ry = r["x"], r["y"]
        if lx == 0 and ly == 0 and rx == 0 and ry == 0:
            midpoints.append(last_mid)
        else:
            mid = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            last_mid = mid
            midpoints.append(mid)

    jumps = []
    for i in range(1, len(midpoints)):
        if midpoints[i] is None or midpoints[i - 1] is None:
            jumps.append(0.0)
        else:
            dx = midpoints[i][0] - midpoints[i - 1][0]
            dy = midpoints[i][1] - midpoints[i - 1][1]
            jumps.append((dx ** 2 + dy ** 2) ** 0.5)
    return np.array(jumps, dtype=np.float32)


# ── JSON sequence loading ─────────────────────────────────────────────────────

def load_sequence(json_path, feature_type, downsample_factor=1):
    """
    Load a keypoint JSON file and return features + metadata.

    Parameters
    ----------
    json_path         : str | Path
    feature_type      : "optical_flow" | "raw_xy"
    downsample_factor : int
        Keep every N-th frame (1 = no downsampling).
        fps is updated accordingly so optical flow and label alignment
        remain correct (e.g. factor=2 converts 50fps → 25fps).

    Returns
    -------
    features : float32 ndarray of shape (T, 75) or (T, 150), or None
    T_valid  : int   — frame count after downsampling
    fps      : float — effective fps after downsampling
    frames   : list  — raw frame dicts (for TED outlier filtering)
    """
    with open(json_path) as f:
        data = json.load(f)
    frames = data["frames"]
    fps    = float(data["fps"])

    raw = [frame_to_coords(f) for f in frames]

    # Forward-fill invalid frames from the nearest preceding valid frame.
    first_valid = next((c for c in raw if c is not None), None)
    if first_valid is None:
        return None, None, fps, frames

    last_valid = first_valid
    coords_seq = []
    for coords in raw:
        if coords is None:
            coords_seq.append(last_valid)
        else:
            last_valid = coords
            coords_seq.append(coords)

    # Downsample: keep every N-th frame, update fps to match.
    if downsample_factor > 1:
        coords_seq = coords_seq[::downsample_factor]
        frames     = frames[::downsample_factor]
        fps        = fps / downsample_factor

    if len(coords_seq) < 2:
        return None, None, fps, frames

    T_valid = len(coords_seq)

    if feature_type == "optical_flow":
        features = compute_optical_flow(coords_seq, fps)
    else:
        features = compute_raw_xy(coords_seq)

    return features, T_valid, fps, frames


def load_sequence_face(body_json_path, face_json_path, feature_type,
                       downsample_factor=1):
    """
    Load body + face keypoint JSON files and return features + metadata.

    Body and face JSONs must come from the same video (same frame count).
    Shoulder width is computed from body landmarks and used to normalise
    both body and face coordinates.

    Parameters
    ----------
    body_json_path    : str | Path
    face_json_path    : str | Path
    feature_type      : "optical_flow" | "raw_xy"
    downsample_factor : int

    Returns
    -------
    features     : float32 ndarray of shape (T, 167) or (T, 334), or None
    T_valid      : int   — frames after downsampling
    fps          : float — effective fps after downsampling
    body_frames  : list  — raw body frame dicts (for TED outlier filtering)
    """
    with open(body_json_path) as f:
        body_data = json.load(f)
    with open(face_json_path) as f:
        face_data = json.load(f)

    body_frames = body_data["frames"]
    face_frames = face_data["frames"]
    fps         = float(body_data["fps"])
    n_frames    = min(len(body_frames), len(face_frames))

    raw = []
    for i in range(n_frames):
        sw = get_shoulder_width(body_frames[i]["pose"])
        raw.append(frame_to_coords_face(body_frames[i], face_frames[i], sw))

    first_valid = next((c for c in raw if c is not None), None)
    if first_valid is None:
        return None, None, fps, body_frames

    last_valid = first_valid
    coords_seq = []
    for coords in raw:
        if coords is None:
            coords_seq.append(last_valid)
        else:
            last_valid = coords
            coords_seq.append(coords)

    # Align body_frames with coords_seq (both derived from the first n_frames).
    body_frames = body_frames[:n_frames]

    if downsample_factor > 1:
        coords_seq  = coords_seq[::downsample_factor]
        body_frames = body_frames[::downsample_factor]
        fps         = fps / downsample_factor

    if len(coords_seq) < 2:
        return None, None, fps, body_frames

    T_valid = len(coords_seq)

    if feature_type == "optical_flow":
        features = compute_optical_flow(coords_seq, fps)
    else:
        features = compute_raw_xy(coords_seq)

    return features, T_valid, fps, body_frames


# ── Video-level train / val / test split ──────────────────────────────────────

def make_video_split(video_ids, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Randomly assign video IDs to train / val / test.

    Returns
    -------
    dict mapping video_id (str) → "train" | "val" | "test"
    """
    rng = random.Random(seed)
    ids = sorted(video_ids)
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    split_map = {}
    for i, vid in enumerate(ids):
        if i < n_train:
            split_map[vid] = "train"
        elif i < n_train + n_val:
            split_map[vid] = "val"
        else:
            split_map[vid] = "test"
    return split_map


# ── Saving ────────────────────────────────────────────────────────────────────

def save_split(output_dir, split, seqs, labs, metas):
    """
    Save one split's sequences, labels, and metadata to output_dir.

    Files written:
      {split}_sequences.npy  — object array of variable-length feature arrays
      {split}_labels.npy     — object array of variable-length int8 label arrays
      {split}_meta.json      — list of per-sequence metadata dicts

    Prints a summary line with frame counts and signing ratio.
    """
    if not seqs:
        print(f"  [WARN] {split} is empty, skipping.")
        return

    X_obj = np.empty(len(seqs), dtype=object)
    y_obj = np.empty(len(seqs), dtype=object)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        X_obj[i] = s
        y_obj[i] = l

    np.save(os.path.join(output_dir, f"{split}_sequences.npy"), X_obj,
            allow_pickle=True)
    np.save(os.path.join(output_dir, f"{split}_labels.npy"), y_obj,
            allow_pickle=True)
    with open(os.path.join(output_dir, f"{split}_meta.json"), "w") as f:
        json.dump(metas, f)

    total = sum(len(s) for s in seqs)
    pos   = sum(int(l.sum()) for l in labs)
    print(f"  {split}: {len(seqs)} sequences | {total:,} frames | "
          f"signing={pos:,} ({100 * pos / total:.1f}%) "
          f"non-signing={total - pos:,} ({100 * (total - pos) / total:.1f}%)")
