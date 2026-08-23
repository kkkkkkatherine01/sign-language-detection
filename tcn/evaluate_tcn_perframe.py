"""Full evaluation for the TCN model."""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from model_tcn_perframe import TCNDetectorPerFrame
from dataset_perframe import SignSequenceDataset, H5SignSequenceDataset, collate_fn
from torch.utils.data import DataLoader


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default=None)
    p.add_argument("--h5_path",       default=None)
    p.add_argument("--save_dir",      required=True)
    p.add_argument("--input_size",    type=int,   default=75)
    p.add_argument("--n_channels",    type=int,   default=32)
    p.add_argument("--kernel_size",   type=int,   default=3)
    p.add_argument("--dilations",     type=int,   nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--num_workers",   type=int,   default=4)
    return p.parse_args()


def get_device():
    torch.backends.cudnn.enabled = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Inference ─────────────────────────────────────────────────────────────────

def get_predictions_with_meta(model, dataset, meta, device, batch_size=4):
    """
    Run model and attach per-frame source + sequence id from meta.

    seq_id groups frames by original video (index into the test split, in
    dataset order since shuffle=False).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    all_probs   = []
    all_labels  = []
    all_sources = []
    all_seq_ids = []

    model.eval()
    seq_idx = 0

    with torch.no_grad():
        for X, y, mask, lengths in loader:
            B = X.shape[0]
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            logits = model(X)
            probs  = torch.sigmoid(logits)

            for b in range(B):
                m      = mask[b]
                T_real = m.sum().item()
                source = meta[seq_idx]["source"] if seq_idx < len(meta) else "unknown"
                all_probs.append(probs[b][m].cpu().numpy())
                all_labels.append(y[b][m].cpu().numpy().astype(int))
                all_sources.extend([source] * T_real)
                all_seq_ids.extend([seq_idx] * T_real)
                seq_idx += 1

    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.array(all_sources),
            np.array(all_seq_ids))


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_training_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per Epoch"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per Epoch"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def save_confusion_matrix(labels, preds, out_dir):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-signing", "Signing"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def save_roc_curve(labels, probs, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")
    return roc_auc


def save_prob_distribution(probs, labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (cls, name) in zip(axes, [(0, "Non-signing"), (1, "Signing")]):
        mask = labels == cls
        ax.hist(probs[mask], bins=50, color="#4C72B0", alpha=0.8)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Frame count")
        ax.set_title(f"Predicted prob distribution — {name}")
        ax.axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "prob_distribution.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def save_per_dataset_breakdown(labels, preds, sources, out_dir):
    datasets = np.unique(sources)
    accs = {}
    for ds in datasets:
        mask = sources == ds
        acc  = (preds[mask] == labels[mask]).mean()
        accs[ds] = acc
        n_sign  = labels[mask].sum()
        n_total = mask.sum()
        print(f"  {ds}: acc={acc:.4f} | {n_total:,} frames "
              f"(signing={n_sign:,} {100*n_sign/n_total:.1f}%)")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(list(accs.keys()), list(accs.values()),
                  color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Dataset Frame Accuracy (Test Set)")
    for bar, val in zip(bars, accs.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    path = os.path.join(out_dir, "per_dataset_accuracy.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    device   = get_device()
    plot_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    history_path = os.path.join(args.save_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        save_training_curves(history, plot_dir)
    else:
        print("history.json not found, skipping training curves.")

    model = TCNDetectorPerFrame(
        input_size=args.input_size,
        n_channels=args.n_channels,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
        input_dropout=0.0,
        block_dropout=0.0,
    ).to(device)
    ckpt = os.path.join(args.save_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"Loaded model from {ckpt}")

    if not args.h5_path and not args.data_dir:
        raise ValueError("Provide either --h5_path or --data_dir")

    if args.h5_path:
        test_ds = H5SignSequenceDataset(args.h5_path, "test")
        meta    = test_ds.meta
    else:
        test_ds   = SignSequenceDataset(args.data_dir, "test")
        meta_path = os.path.join(args.data_dir, "test_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

    print("\nRunning inference on test set...")
    probs, labels, sources, seq_ids = get_predictions_with_meta(
        model, test_ds, meta, device, args.batch_size)
    preds = (probs >= args.threshold).astype(int)

    pred_path = os.path.join(args.save_dir, "test_predictions.npz")
    np.savez(
        pred_path,
        probs=probs,
        labels=labels,
        preds=preds,
        sources=sources,
        seq_ids=seq_ids,
    )
    print(f"Saved per-frame predictions for bootstrap to: {pred_path}")

    print("\n=== Classification Report ===")
    report = classification_report(
        labels, preds,
        target_names=["Non-signing", "Signing"],
        digits=4,
    )
    print(report)
    with open(os.path.join(args.save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    save_confusion_matrix(labels, preds, plot_dir)
    roc_auc = save_roc_curve(labels, probs, plot_dir)
    print(f"AUC: {roc_auc:.4f}")
    save_prob_distribution(probs, labels, plot_dir)

    print("\n=== Per-Dataset Breakdown ===")
    save_per_dataset_breakdown(labels, preds, sources, plot_dir)

    print(f"\nDone. All outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
