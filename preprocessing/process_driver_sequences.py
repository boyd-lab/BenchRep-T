"""
Process VDJdb ground truth data and find matching sequences in Mal-ID TCR data.

For each disease (Covid19, HIV, Influenza):
  1. Load VDJdb ground truth, filter to Score >= 2, validate and trim CDR3 sequences.
  2. Find matching Mal-ID specimens for that disease from metadata.
  3. For each specimen, find cdr3_aa sequences with >= 90% Levenshtein similarity
     to any ground truth sequence.
  4. Write a per-specimen output file listing matched sequences.

The three diseases are processed in parallel (one process each).
"""

import argparse
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import Levenshtein
from tqdm import tqdm


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
    tqdm.write(f"[{disease}] {len(df)} sequences after Score >= {MIN_SCORE} filter")

    trimmed = []
    for idx, row in df.iterrows():
        seq = str(row["CDR3"])
        if not seq.startswith("C"):
            tqdm.write(f"  WARNING [{disease}] CDR3 at row {idx} does not start with 'C': {seq!r}")
        if not seq.endswith("F"):
            tqdm.write(f"  WARNING [{disease}] CDR3 at row {idx} does not end with 'F': {seq!r}")
        # Only trim the terminal residues if present
        if seq.startswith("C"):
            seq = seq[1:]
        if seq.endswith("F"):
            seq = seq[:-1]
        trimmed.append(seq)

    tqdm.write(f"[{disease}] {len(trimmed)} ground truth sequences loaded")
    return trimmed


# ---------------------------------------------------------------------------
# Per-disease worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def process_disease(disease: str, vdjdb_path: str, output_dir: str, tqdm_position: int) -> str:
    """Process one disease. Returns a summary string."""
    tqdm.write(f"\n[{disease}] Starting (VDJdb: {vdjdb_path})")

    ground_truth = load_ground_truth(disease, vdjdb_path)
    if not ground_truth:
        return f"[{disease}] WARNING: no ground truth sequences, skipped."

    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    specimens = (
        metadata[metadata["disease"] == disease][["participant_label", "specimen_label"]]
        .drop_duplicates()
    )
    tqdm.write(f"[{disease}] Found {len(specimens)} specimens")

    if specimens.empty:
        return f"[{disease}] WARNING: no specimens found in metadata."

    disease_output_dir = os.path.join(output_dir, disease)
    os.makedirs(disease_output_dir, exist_ok=True)

    n_matched_specimens = 0
    spec_bar = tqdm(
        list(specimens.iterrows()),
        desc=f"{disease} specimens",
        unit="specimen",
        position=tqdm_position,
        leave=True,
        dynamic_ncols=True,
    )
    for _, spec_row in spec_bar:
        participant = spec_row["participant_label"]
        specimen    = spec_row["specimen_label"]
        tcr_file    = os.path.join(TCR_DIR, f"part_table_{participant}_{specimen}.tsv.gz")

        spec_bar.set_postfix(specimen=specimen)

        if not os.path.exists(tcr_file):
            tqdm.write(f"  WARNING [{disease}] TCR file not found, skipping: {tcr_file}")
            continue

        with gzip.open(tcr_file, "rt") as fh:
            tcr_df = pd.read_csv(fh, sep="\t", low_memory=False)

        if "cdr3_aa" not in tcr_df.columns:
            tqdm.write(f"  WARNING [{disease}] 'cdr3_aa' column missing in {tcr_file}, skipping.")
            continue

        unique_cdr3s = tcr_df["cdr3_aa"].dropna().unique().tolist()

        matched = []
        for cdr3 in tqdm(
            unique_cdr3s,
            desc=f"  {specimen} cdr3s",
            unit="seq",
            position=tqdm_position + 1,
            leave=False,
            dynamic_ncols=True,
        ):
            cdr3_str = str(cdr3)
            for gt_seq in ground_truth:
                if seq_similarity(cdr3_str, gt_seq) >= SIMILARITY_THRESHOLD:
                    matched.append(cdr3_str)
                    break  # only one GT hit needed per cdr3

        if matched:
            out_path = os.path.join(disease_output_dir, f"{specimen}.tsv")
            pd.DataFrame({"cdr3_aa": matched}).to_csv(out_path, sep="\t", index=False)
            tqdm.write(f"  [{disease}] {specimen}: {len(matched)} matched sequences -> {out_path}")
            n_matched_specimens += 1
        else:
            tqdm.write(f"  [{disease}] {specimen}: no matches")

    return f"[{disease}] Done. {n_matched_specimens}/{len(specimens)} specimens had matches."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Validate metadata columns up front before spawning workers
    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    required_cols = {"participant_label", "specimen_label", "disease"}
    missing = required_cols - set(metadata.columns)
    if missing:
        sys.exit(f"ERROR: metadata is missing columns: {missing}")

    # Each disease gets its own tqdm row (position 0, 2, 4) with the inner
    # cdr3 bar on the row below (position 1, 3, 5), leaving no overlap.
    jobs = [
        (disease, vdjdb_path, output_dir, idx * 2)
        for idx, (disease, vdjdb_path) in enumerate(VDJDB_FILES.items())
    ]

    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(process_disease, disease, vdjdb_path, output_dir, pos): disease
            for disease, vdjdb_path, output_dir, pos in jobs
        }
        for future in as_completed(futures):
            disease = futures[future]
            try:
                summary = future.result()
                tqdm.write(summary)
            except Exception as exc:
                tqdm.write(f"ERROR [{disease}]: {exc}")

    print("\nAll diseases complete.")


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
