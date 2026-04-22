"""
Compute evaluation metrics (AUROC, AUPR, F1, balanced accuracy) from an
evaluation output CSV produced by the scripts in ``evals/``.

Expected columns:
    - disease_label (0/1)
    - model_score (float in [0, 1])
    - malid_cross_validation_fold_id_when_in_test_set (int)

Reports per-fold metrics (mean ± std across folds) and overall metrics
computed by concatenating predictions from all folds.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"
LABEL_COL = "disease_label"
SCORE_COL = "model_score"


def _compute(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(labels, scores),
        "aupr": average_precision_score(labels, scores),
        "f1": f1_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "n": len(labels),
        "n_pos": int(labels.sum()),
    }


def compute_metrics(csv_path, threshold=0.5):
    """Return (per_fold_df, overall_dict) for the given output CSV."""
    df = pd.read_csv(csv_path)
    missing = {FOLD_COL, LABEL_COL, SCORE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    fold_rows = []
    for fold, group in df.groupby(FOLD_COL):
        m = _compute(group[LABEL_COL].values, group[SCORE_COL].values, threshold)
        m["fold"] = int(fold)
        fold_rows.append(m)
    per_fold = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)

    overall = _compute(df[LABEL_COL].values, df[SCORE_COL].values, threshold)
    return per_fold, overall


def _print_report(csv_path, per_fold, overall, threshold):
    bar = "=" * 60
    print(bar)
    print(f"Metrics for: {csv_path}")
    print(f"Threshold for F1 / balanced accuracy: {threshold}")
    print(bar)

    print("\nPer-fold metrics:")
    display = per_fold[["fold", "n", "n_pos", "auroc", "aupr", "f1", "balanced_acc"]]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMean ± std across folds:")
    for key, label in [
        ("auroc", "AUROC       "),
        ("aupr", "AUPR        "),
        ("f1", "F1          "),
        ("balanced_acc", "Balanced Acc"),
    ]:
        vals = per_fold[key].values
        print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("\nOverall (all folds concatenated):")
    print(f"  n = {overall['n']}  (positives: {overall['n_pos']})")
    print(f"  AUROC       : {overall['auroc']:.4f}")
    print(f"  AUPR        : {overall['aupr']:.4f}")
    print(f"  F1          : {overall['f1']:.4f}")
    print(f"  Balanced Acc: {overall['balanced_acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to evaluation output CSV.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for F1 and balanced accuracy (default: 0.5).",
    )
    args = parser.parse_args()

    per_fold, overall = compute_metrics(args.csv_path, threshold=args.threshold)
    _print_report(args.csv_path, per_fold, overall, args.threshold)


if __name__ == "__main__":
    main()
