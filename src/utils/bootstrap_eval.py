import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", required=True,
                   help="Path to test_predictions.npz")
    p.add_argument("--compare_path", default=None,
                   help="Optional second test_predictions.npz, to run a "
                        "paired bootstrap significance test against --pred_path. "
                        "Must have been evaluated on the SAME test set / same "
                        "seq_id ordering (e.g. LSTM vs TCN on the same H5 file).")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_path", default=None,
                   help="Optional path to dump results as JSON "
                        "(default: alongside --pred_path)")
    return p.parse_args()


# ── Core bootstrap ────────────────────────────────────────────────────────────

def block_bootstrap_accuracy(labels, preds, seq_ids, n_boot=1000, seed=42):
    """
    Resample whole sequences (videos) with replacement, n_boot times,
    and compute test accuracy on each resampled set.
    """
    rng = np.random.default_rng(seed)
    unique_seqs = np.unique(seq_ids)
    n_seqs = len(unique_seqs)
    seq_to_indices = {s: np.where(seq_ids == s)[0] for s in unique_seqs}

    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique_seqs, size=n_seqs, replace=True)
        idx = np.concatenate([seq_to_indices[s] for s in sampled])
        boot_scores[i] = (preds[idx] == labels[idx]).mean()

    return {
        "bootstrap_mean": float(boot_scores.mean()),
        "bootstrap_std": float(boot_scores.std(ddof=1)),
        "ci_95": [float(np.percentile(boot_scores, 2.5)),
                  float(np.percentile(boot_scores, 97.5))],
        "n_sequences": int(n_seqs),
        "point_estimate": float((preds == labels).mean()),
    }


def paired_block_bootstrap_diff(labels, preds_a, preds_b, seq_ids, n_boot=1000, seed=42):
    """
    Same resampling draw used for both configs at once, so the comparison is
    apples-to-apples. If the 95% CI of the difference excludes 0, the
    difference is significant at the 5% level.
    """
    rng = np.random.default_rng(seed)
    unique_seqs = np.unique(seq_ids)
    n_seqs = len(unique_seqs)
    seq_to_indices = {s: np.where(seq_ids == s)[0] for s in unique_seqs}

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique_seqs, size=n_seqs, replace=True)
        idx = np.concatenate([seq_to_indices[s] for s in sampled])
        acc_a = (preds_a[idx] == labels[idx]).mean()
        acc_b = (preds_b[idx] == labels[idx]).mean()
        diffs[i] = acc_a - acc_b

    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "significant_at_5pct": bool(ci_lower > 0 or ci_upper < 0),
    }


def per_source_bootstrap(labels, preds, sources, seq_ids, n_boot, seed):
    results = {}
    for src in np.unique(sources):
        mask = sources == src
        results[str(src)] = block_bootstrap_accuracy(
            labels[mask], preds[mask], seq_ids[mask], n_boot=n_boot, seed=seed
        )
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data = np.load(args.pred_path)
    labels, preds, sources, seq_ids = (
        data["labels"], data["preds"], data["sources"], data["seq_ids"]
    )
    print(f"Loaded {len(labels):,} frames from {len(np.unique(seq_ids))} sequences "
          f"({args.pred_path})")

    overall = block_bootstrap_accuracy(labels, preds, seq_ids,
                                        n_boot=args.n_boot, seed=args.seed)
    print("\n=== Overall (block bootstrap by sequence) ===")
    print(f"Point estimate accuracy: {overall['point_estimate']:.4f}")
    print(f"Bootstrap mean ± std:    {overall['bootstrap_mean']:.4f} ± {overall['bootstrap_std']:.4f}")
    print(f"95% CI:                  [{overall['ci_95'][0]:.4f}, {overall['ci_95'][1]:.4f}]")

    results = {"overall": overall}

    unique_sources = np.unique(sources)
    if len(unique_sources) > 1:
        print("\n=== Per-source breakdown ===")
        per_source = per_source_bootstrap(labels, preds, sources, seq_ids,
                                          args.n_boot, args.seed)
        results["per_source"] = per_source
        for src, r in per_source.items():
            print(f"  {src}: {r['bootstrap_mean']:.4f} ± {r['bootstrap_std']:.4f} "
                  f"| 95% CI [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}] "
                  f"| n_seq={r['n_sequences']}")

    if args.compare_path:
        data_b = np.load(args.compare_path)
        labels_b, preds_b, seq_ids_b = data_b["labels"], data_b["preds"], data_b["seq_ids"]

        if len(labels) != len(labels_b) or not np.array_equal(seq_ids, seq_ids_b):
            print("\n[WARNING] --pred_path and --compare_path do not appear to be "
                  "evaluated on the exact same frames/sequence ordering — paired "
                  "comparison assumes they are. Results below may not be meaningful.")

        diff = paired_block_bootstrap_diff(labels, preds, preds_b, seq_ids,
                                           n_boot=args.n_boot, seed=args.seed)
        print(f"\n=== Paired comparison: {os.path.basename(args.pred_path)} "
              f"vs {os.path.basename(args.compare_path)} ===")
        print(f"Mean accuracy difference (A - B): {diff['mean_diff']:.4f}")
        print(f"95% CI of difference:             [{diff['ci_95'][0]:.4f}, {diff['ci_95'][1]:.4f}]")
        print(f"Significant at 5% level:          {diff['significant_at_5pct']}")
        results["paired_comparison"] = diff

    out_path = args.out_path or os.path.join(
        os.path.dirname(args.pred_path), "bootstrap_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out_path}")


if __name__ == "__main__":
    main()
