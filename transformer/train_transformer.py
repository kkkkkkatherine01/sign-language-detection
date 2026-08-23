"""
train_transformer.py

Per-frame Transformer training script.
Mirrors train_perframe.py — only the model and its args differ, to keep
the LSTM/Transformer comparison fair.

Usage:
    python train_transformer.py \
        --h5_path  /path/to/dataset.h5 \
        --save_dir /path/to/checkpoints/transformer_bh \
        --input_size 75
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
from model_transformer import TransformerDetectorPerFrame
from dataset_perframe import get_dataloaders, get_dataloaders_h5, collate_fn


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_dir",          default=None,
                   help="Directory with {split}_sequences.npy files")
    p.add_argument("--h5_path",           default=None,
                   help="Packed HDF5 file (overrides --data_dir)")
    p.add_argument("--save_dir",          required=True)

    # Model — Transformer-specific
    p.add_argument("--input_size",        type=int,   default=75)
    p.add_argument("--d_model",           type=int,   default=64,
                   help="Transformer d_model (analogous to LSTM hidden_size=64)")
    p.add_argument("--nhead",             type=int,   default=4,
                   help="Number of attention heads (d_model must be divisible)")
    p.add_argument("--num_encoder_layers",type=int,   default=2,
                   help="Number of TransformerEncoderLayer stacked")
    p.add_argument("--dim_feedforward",   type=int,   default=256,
                   help="FFN hidden dim inside each encoder layer")
    p.add_argument("--dropout",           type=float, default=0.1,
                   help="Attention + FFN dropout")
    p.add_argument("--input_dropout",     type=float, default=0.1,
                   help="Input feature dropout (same as LSTM baseline)")
    p.add_argument("--window_size",       type=int,   default=50,
                   help="Sliding window size in frames (50 = 2s @25fps)")

    # Training — identical to train_perframe.py
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--batch_size",        type=int,   default=8)
    p.add_argument("--val_batch_size",    type=int,   default=4)
    p.add_argument("--steps_per_epoch",   type=int,   default=256,
                   help="Batches per epoch. -1 = full dataset pass.")
    p.add_argument("--lr",                type=float, default=3e-4,
                   help="Learning rate. 3e-4 recommended for Transformers (vs 1e-3 for LSTM).")
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--patience",          type=int,   default=5)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--num_gpus",          type=int,   default=1,
                   help="Number of GPUs via DataParallel (1 = single GPU)")
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(num_gpus: int = 1):
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        n_avail = torch.cuda.device_count()
        n_use   = min(num_gpus, n_avail)
        print(f"Available GPUs: {n_avail} | Using: {n_use}")
        for i in range(n_avail):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda"), n_use
    print("Using CPU")
    return torch.device("cpu"), 1


def infinite_loader(loader):
    """Cycle through a DataLoader indefinitely (for steps_per_epoch)."""
    while True:
        for batch in loader:
            yield batch


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, inf_loader, criterion, optimizer, device, steps):
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_frames  = 0

    for _ in range(steps):
        X, y, mask, lengths = next(inf_loader)
        X, y, mask = X.to(device), y.to(device), mask.to(device)

        logits = model(X, lengths)                    # (batch, T)
        loss   = criterion(logits[mask], y[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            preds   = (torch.sigmoid(logits[mask]) >= 0.5).float()
            correct = (preds == y[mask]).sum().item()
            n       = mask.sum().item()

        total_loss    += loss.item() * n
        total_correct += correct
        total_frames  += n

    if total_frames == 0:
        return float('nan'), 0.0
    return total_loss / total_frames, total_correct / total_frames


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_frames  = 0

    with torch.no_grad():
        for X, y, mask, lengths in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            logits  = model(X, lengths)
            loss    = criterion(logits[mask], y[mask])
            preds   = (torch.sigmoid(logits[mask]) >= 0.5).float()
            correct = (preds == y[mask]).sum().item()
            n       = mask.sum().item()
            total_loss    += loss.item() * n
            total_correct += correct
            total_frames  += n

    return total_loss / total_frames, total_correct / total_frames


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device, num_gpus = get_device(args.num_gpus)

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if not args.h5_path and not args.data_dir:
        raise ValueError("Provide either --h5_path or --data_dir")

    # Data
    if args.h5_path:
        train_loader, val_loader, test_loader, _ = get_dataloaders_h5(
            args.h5_path, args.batch_size, args.num_workers)
    else:
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers)

    steps = len(train_loader) if args.steps_per_epoch == -1 else args.steps_per_epoch
    print(f"steps_per_epoch = {steps}")

    # Sliding window limits attention to O(W²) — val/test can use larger batch_size.
    pin = torch.cuda.is_available()
    val_loader = DataLoader(
        val_loader.dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_loader.dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn,
    )

    inf_train = infinite_loader(train_loader)

    # Model
    model = TransformerDetectorPerFrame(
        input_size         = args.input_size,
        d_model            = args.d_model,
        nhead              = args.nhead,
        num_encoder_layers = args.num_encoder_layers,
        dim_feedforward    = args.dim_feedforward,
        dropout            = args.dropout,
        input_dropout      = args.input_dropout,
        window_size        = args.window_size,
    ).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Keep a reference to the unwrapped model for state_dict ops
    base_model = model

    # Loss with pos_weight (same as LSTM)
    train_labels = train_loader.dataset.labels
    n_pos = sum(int(l.sum()) for l in train_labels)
    n_neg = sum(len(l) - int(l.sum()) for l in train_labels)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    print(f"pos_weight = {pos_weight.item():.3f} (neg={n_neg:,} pos={n_pos:,})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=3, factor=0.5)

    best_ckpt    = os.path.join(args.save_dir, "best_model.pt")
    resume_ckpt  = os.path.join(args.save_dir, "checkpoint_latest.pt")

    # ── Resume from checkpoint if available ──────────────────────────────────
    history          = {"train_loss": [], "val_loss": [],
                        "train_acc":  [], "val_acc":  []}
    best_val_acc     = 0.0
    patience_counter = 0
    start_epoch      = 1

    if os.path.exists(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history          = ckpt["history"]
        best_val_acc     = ckpt["best_val_acc"]
        patience_counter = ckpt["patience_counter"]
        start_epoch      = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch} | "
              f"best_val_acc={best_val_acc:.4f} | "
              f"patience_counter={patience_counter}")
    else:
        print("No checkpoint found, training from scratch.")

    # Wrap with DataParallel AFTER checkpoint restore (base_model weights are already loaded)
    if num_gpus > 1:
        model = nn.DataParallel(base_model, device_ids=list(range(num_gpus)))
        print(f"DataParallel enabled on {num_gpus} GPUs")
    else:
        model = base_model

    print("=== Training ===")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, inf_train, criterion, optimizer, device, steps)
        val_loss, val_acc = eval_epoch(
            base_model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(base_model.state_dict(), best_ckpt)
            print(f"  ✓ saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                torch.save({
                    "epoch":           epoch,
                    "model":           base_model.state_dict(),
                    "optimizer":       optimizer.state_dict(),
                    "scheduler":       scheduler.state_dict(),
                    "history":         history,
                    "best_val_acc":    best_val_acc,
                    "patience_counter":patience_counter,
                }, resume_ckpt)
                break

        # Save latest checkpoint every epoch (overwrites previous)
        torch.save({
            "epoch":           epoch,
            "model":           base_model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "history":         history,
            "best_val_acc":    best_val_acc,
            "patience_counter":patience_counter,
        }, resume_ckpt)

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("History saved.")

    # Test
    print("\n=== Test Set Evaluation ===")
    base_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = eval_epoch(base_model, test_loader, criterion, device)
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.4f}")
    print("Run evaluate_transformer.py for full metrics.")


if __name__ == "__main__":
    main()
