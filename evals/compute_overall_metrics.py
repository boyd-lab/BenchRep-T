"""
Recompute overall metrics from result CSVs without re-running experiments.

The "Overall" metrics printed at the end of each log are computed by
concatenating all per-fold test predictions and scoring once — identical
to calling the sklearn functions on the full `disease_label` / `model_score`
columns in the CSV.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(df: pd.DataFrame) -> dict:
    y = df["disease_label"].values
    p = df["model_score"].values
    preds = (p >= 0.5).astype(int)
    return {
        "overall_auroc": roc_auc_score(y, p),
        "overall_aupr": average_precision_score(y, p),
        "overall_balanced_acc": balanced_accuracy_score(y, preds),
        "overall_f1": f1_score(y, preds),
        "n_disease": int(y.sum()),
        "n_healthy": int((y == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute overall metrics from result CSVs"
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="CSV file(s) to process. Accepts globs. "
             "Defaults to airr_bench/results/*.csv relative to this script.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save summary table to this CSV path (optional)",
    )
    args = parser.parse_args()

    if args.csv_paths:
        paths = []
        for p in args.csv_paths:
            paths.extend(glob.glob(p))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths = glob.glob(os.path.join(script_dir, "results", "*.csv"))

    if not paths:
        sys.exit("No CSV files found.")

    paths = sorted(paths)
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        metrics = compute_metrics(df)
        metrics["file"] = os.path.basename(path)
        rows.append(metrics)
        print(
            f"{metrics['file']}\n"
            f"  AUROC={metrics['overall_auroc']:.4f}  "
            f"AUPR={metrics['overall_aupr']:.4f}  "
            f"BalAcc={metrics['overall_balanced_acc']:.4f}  "
            f"F1={metrics['overall_f1']:.4f}  "
            f"(n_disease={metrics['n_disease']}, n_healthy={metrics['n_healthy']})\n"
        )

    summary = pd.DataFrame(rows)[
        ["file", "overall_auroc", "overall_aupr", "overall_balanced_acc", "overall_f1",
         "n_disease", "n_healthy"]
    ]
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
