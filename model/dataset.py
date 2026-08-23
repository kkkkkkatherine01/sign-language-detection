"""DataLoader for per-frame variable-length sequences.
collate_fn pads sequences and returns lengths for pack_padded_sequence.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SignSequenceDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.seqs   = np.load(os.path.join(data_dir, f"{split}_sequences.npy"),
                              allow_pickle=True)
        self.labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"),
                              allow_pickle=True)

        total = sum(len(l) for l in self.labels)
        pos   = sum(int(l.sum()) for l in self.labels)
        print(f"[{split}] {len(self.seqs)} sequences | {total:,} frames | "
              f"signing={pos:,} ({100*pos/total:.1f}%)")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        X = torch.tensor(self.seqs[idx],   dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y


def collate_fn(batch):
    Xs, ys   = zip(*batch)
    lengths  = torch.tensor([x.shape[0] for x in Xs], dtype=torch.int64)
    max_T    = lengths.max().item()
    feat_dim = Xs[0].shape[1]
    B        = len(Xs)

    X_pad = torch.zeros(B, max_T, feat_dim)
    y_pad = torch.zeros(B, max_T)
    mask  = torch.zeros(B, max_T, dtype=torch.bool)

    for i, (x, y, T) in enumerate(zip(Xs, ys, lengths.tolist())):
        X_pad[i, :T] = x
        y_pad[i, :T] = y
        mask[i, :T]  = True

    return X_pad, y_pad, mask, lengths


def get_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    train_ds = SignSequenceDataset(data_dir, "train")
    val_ds   = SignSequenceDataset(data_dir, "val")
    test_ds  = SignSequenceDataset(data_dir, "test")

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, test_ds


# ── HDF5-backed dataset (single-file format) ──────────────────────────────────

class H5SignSequenceDataset(Dataset):
    def __init__(self, h5_path: str, split: str):
        import h5py
        self.h5_path = h5_path
        self.split   = split

        with h5py.File(h5_path, "r") as f:
            grp          = f[split]
            self.lengths = grp["lengths"][:]
            labs_flat    = grp["labels"][:]
            meta_json    = grp.attrs.get("meta", "[]")

        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.labels  = [labs_flat[self.offsets[i]:self.offsets[i + 1]]
                        for i in range(len(self.lengths))]
        self.meta    = json.loads(meta_json)
        self._h5     = None   # opened lazily in each worker process

        total = int(self.lengths.sum())
        pos   = sum(int(l.sum()) for l in self.labels)
        print(f"[{split}] {len(self.lengths)} sequences | {total:,} frames | "
              f"signing={pos:,} ({100 * pos / total:.1f}%)")

    def __len__(self):
        return len(self.lengths)

    def _open(self):
        """Return a worker-local h5py file handle."""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        s = int(self.offsets[idx])
        e = int(self.offsets[idx + 1])
        X = torch.tensor(self._open()[f"{self.split}/sequences"][s:e],
                         dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def get_dataloaders_h5(h5_path: str, batch_size: int = 16, num_workers: int = 4):
    train_ds = H5SignSequenceDataset(h5_path, "train")
    val_ds   = H5SignSequenceDataset(h5_path, "val")
    test_ds  = H5SignSequenceDataset(h5_path, "test")

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, test_ds
