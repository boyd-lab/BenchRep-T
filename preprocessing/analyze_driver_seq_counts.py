"""
Analyze the number of unique driver sequences (sample_cdr3) discovered per specimen
for Covid19, HIV, and Influenza. Specimens with 0 matches are included by using
metadata.tsv to enumerate all specimens for each disease.

Usage:
    conda activate airr-bench
    python airr_bench/preprocessing/analyze_driver_seq_counts.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VDJDB_MATCHES = REPO_ROOT / "airr_bench/data/vdjdb_matches.csv"
METADATA = REPO_ROOT / "airr_bench/data/malid_clean/metadata.tsv"
OUT_DIR = REPO_ROOT / "airr_bench/preprocessing/reports"

DISEASES_TO_PLOT = ["Covid19", "HIV", "Influenza"]


def extract_specimen_label(filename: str) -> str:
    """Extract specimen_label from filename like 'part_table_BFI-0007450_M369-S001'."""
    # Format: part_table_{participant_label}_{specimen_label}
    return filename.rsplit("_", 1)[-1]


def load_driver_counts(matches_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns [specimen_label, n_driver_seqs].
    Counts unique sample_cdr3 per specimen across all vdjdb_matches rows.
    """
    df = pd.read_csv(matches_path)
    df["specimen_label"] = df["filename"].apply(extract_specimen_label)

    counts = (
        df.groupby("specimen_label")["sample_cdr3"]
        .nunique()
        .reset_index(name="n_driver_seqs")
    )
    return counts


def load_specimens_by_disease(metadata_path: Path, diseases: list[str]) -> pd.DataFrame:
    """
    Return all specimens from metadata whose disease is in `diseases`.
    This is the ground-truth population for computing 0-match specimens.
    """
    meta = pd.read_csv(metadata_path, sep="\t")
    filtered = meta.loc[meta["disease"].isin(diseases), ["specimen_label", "disease"]]
    return filtered.drop_duplicates()


def main(matches_path: Path, metadata_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    driver_counts = load_driver_counts(matches_path)
    # Only include specimens whose disease is one of the target diseases
    all_specimens = load_specimens_by_disease(metadata_path, DISEASES_TO_PLOT)

    # Left-join: specimens absent from vdjdb_matches get 0
    merged = all_specimens.merge(driver_counts, on="specimen_label", how="left")
    merged["n_driver_seqs"] = merged["n_driver_seqs"].fillna(0).astype(int)

    print(merged.groupby("disease")["n_driver_seqs"].describe().to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, disease in zip(axes, DISEASES_TO_PLOT):
        subset = merged.loc[merged["disease"] == disease, "n_driver_seqs"]
        n_specimens = len(subset)
        n_zero = (subset == 0).sum()

        ax.hist(subset, bins=30, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{disease}\n(n={n_specimens}, {n_zero} with 0 matches)")
        ax.set_xlabel("Unique driver sequences (sample_cdr3)")
        ax.set_ylabel("Number of specimens")

    fig.suptitle("Driver sequence counts per specimen by disease", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / "driver_seq_counts_by_disease.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    summary_path = out_dir / "driver_seq_counts_per_specimen.csv"
    merged.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=VDJDB_MATCHES)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(args.matches, args.metadata, args.out_dir)
