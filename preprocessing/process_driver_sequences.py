"""
Process VDJdb ground truth data and find matching sequences in Mal-ID TCR data.

For each disease (Covid19, HIV, Influenza):
  1. Load VDJdb ground truth, filter to Score >= 2, validate and trim CDR3 sequences.
  2. Find matching Mal-ID specimens for that disease from metadata.
  3. For each specimen, find cdr3_aa sequences with >= 90% Levenshtein similarity
     to any ground truth sequence.
  4. Write a per-specimen output file listing matched sequences.
"""

import argparse
import gzip
import os
import sys

import pandas as pd
import Levenshtein


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # airr_bench/

VDJDB_FILES = {
    "Covid19":   os.path.join(REPO_ROOT, "data", "public_clones", "Covid19_GT_TRB3.tsv"),
    "HIV":       os.path.join(REPO_ROOT, "data", "public_clones", "HIV_GT_TRB3.tsv"),
    "Influenza": os.path.join(REPO_ROOT, "data", "public_clones", "Influenza_GT_TRB3.tsv"),
}

METADATA_PATH = os.path.join(REPO_ROOT, "data", "malid_clean", "metadata.tsv")
TCR_DIR       = os.path.join(REPO_ROOT, "data", "malid_clean", "TCR")

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "vdjdb_matches")

SIMILARITY_THRESHOLD = 0.90
MIN_SCORE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seq_similarity(s1: str, s2: str) -> float:
    """Similarity in [0, 1]: 1 - normalised Levenshtein distance."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    return 1.0 - Levenshtein.distance(s1, s2) / max_len


def load_ground_truth(disease: str, vdjdb_path: str) -> list:
    """
    Load VDJdb file, filter Score >= MIN_SCORE, validate CDR3 starts with C
    and ends with F, then strip those terminal residues.
    Returns a list of trimmed CDR3 strings.
    """
    df = pd.read_csv(vdjdb_path, sep="\t")

    for col in ("Score", "CDR3"):
        if col not in df.columns:
            sys.exit(f"ERROR: '{col}' column not found in {vdjdb_path}")

    df = df[df["Score"] >= MIN_SCORE].copy()
    print(f"  [{disease}] {len(df)} sequences after Score >= {MIN_SCORE} filter")

    trimmed = []
    for idx, row in df.iterrows():
        seq = str(row["CDR3"])
        if not seq.startswith("C"):
            sys.exit(
                f"ERROR: CDR3 at row {idx} in {vdjdb_path} does not start with 'C': {seq!r}"
            )
        if not seq.endswith("F"):
            sys.exit(
                f"ERROR: CDR3 at row {idx} in {vdjdb_path} does not end with 'F': {seq!r}"
            )
        trimmed.append(seq[1:-1])  # strip leading C and trailing F

    print(f"  [{disease}] {len(trimmed)} ground truth sequences loaded")
    return trimmed


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    required_cols = {"participant_label", "specimen_label", "disease"}
    missing = required_cols - set(metadata.columns)
    if missing:
        sys.exit(f"ERROR: metadata is missing columns: {missing}")

    for disease, vdjdb_path in VDJDB_FILES.items():
        print(f"\nProcessing disease: {disease}")
        print(f"  VDJdb file: {vdjdb_path}")

        ground_truth = load_ground_truth(disease, vdjdb_path)
        if not ground_truth:
            print(f"  WARNING: no ground truth sequences for {disease}, skipping.")
            continue

        # Find all specimens for this disease in metadata
        specimens = (
            metadata[metadata["disease"] == disease][["participant_label", "specimen_label"]]
            .drop_duplicates()
        )
        print(f"  Found {len(specimens)} specimens for {disease}")

        if specimens.empty:
            print(f"  WARNING: no specimens found for disease '{disease}' in metadata.")
            continue

        disease_output_dir = os.path.join(output_dir, disease)
        os.makedirs(disease_output_dir, exist_ok=True)

        for _, spec_row in specimens.iterrows():
            participant = spec_row["participant_label"]
            specimen    = spec_row["specimen_label"]
            tcr_file    = os.path.join(TCR_DIR, f"part_table_{participant}_{specimen}.tsv.gz")

            if not os.path.exists(tcr_file):
                print(f"  WARNING: TCR file not found, skipping: {tcr_file}")
                continue

            with gzip.open(tcr_file, "rt") as fh:
                tcr_df = pd.read_csv(fh, sep="\t", low_memory=False)

            if "cdr3_aa" not in tcr_df.columns:
                print(f"  WARNING: 'cdr3_aa' column missing in {tcr_file}, skipping.")
                continue

            unique_cdr3s = tcr_df["cdr3_aa"].dropna().unique().tolist()

            matched = []
            for cdr3 in unique_cdr3s:
                cdr3_str = str(cdr3)
                for gt_seq in ground_truth:
                    if seq_similarity(cdr3_str, gt_seq) >= SIMILARITY_THRESHOLD:
                        matched.append(cdr3_str)
                        break  # only one GT hit needed per cdr3

            if matched:
                out_path = os.path.join(disease_output_dir, f"{specimen}.tsv")
                pd.DataFrame({"cdr3_aa": matched}).to_csv(out_path, sep="\t", index=False)
                print(f"  {specimen}: {len(matched)} matched sequences -> {out_path}")
            else:
                print(f"  {specimen}: no matches")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match Mal-ID TCR cdr3_aa sequences against VDJdb ground truth."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    process(args.output_dir)
