"""TCN training script."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from model_tcn import TCNDetectorPerFrame
from dataset import get_dataloaders, get_dataloaders_h5


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        default=None)
    p.add_argument("--h5_path",         default=None)
    p.add_argument("--save_dir",        required=True)
    p.add_argument("--input_size",      type=int,   default=75)
    p.add_argument("--n_channels",      type=int,   default=48)
    p.add_argument("--kernel_size",     type=int,   default=5)
    p.add_argument("--dilations",       type=int,   nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--input_dropout",   type=float, default=0.5)
    p.add_argument("--block_dropout",   type=float, default=0.2)
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--val_batch_size",  type=int,   default=32)
    p.add_argument("--steps_per_epoch", type=int,   default=-1)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def infinite_loader(loader):
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

        logits = model(X, lengths)
        loss   = criterion(logits[mask], y[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds   = (torch.sigmoid(logits[mask]) >= 0.5).float()
            correct = (preds == y[mask]).sum().item()
            n       = mask.sum().item()

        total_loss    += loss.item() * n
        total_correct += correct
        total_frames  += n

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
    device = get_device()

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if not args.h5_path and not args.data_dir:
        raise ValueError("Provide either --h5_path or --data_dir")

    if args.h5_path:
        train_loader, val_loader, test_loader, _ = get_dataloaders_h5(
            args.h5_path, args.batch_size, args.num_workers)
    else:
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers)

    if args.steps_per_epoch == -1:
        steps = len(train_loader)
        print(f"steps_per_epoch = full dataset ({steps} batches)")
    else:
        steps = args.steps_per_epoch
        print(f"steps_per_epoch = {steps}")

    inf_train = infinite_loader(train_loader)

    # Model
    model = TCNDetectorPerFrame(
        input_size=args.input_size,
        n_channels=args.n_channels,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
        input_dropout=args.input_dropout,
        block_dropout=args.block_dropout,
    ).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss
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

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_acc     = 0.0
    patience_counter = 0
    best_ckpt        = os.path.join(args.save_dir, "best_model.pt")

    print("=== Training ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, inf_train, criterion, optimizer, device, steps)
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device)

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
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("History saved.")

    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.4f}")
    print("Run evaluate_tcn_perframe.py for full metrics.")


if __name__ == "__main__":
    main()
