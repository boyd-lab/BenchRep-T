"""
Process VDJdb ground truth data and find matching sequences in Mal-ID TCR data.

For each disease (Covid19, HIV, Influenza):
  1. Load VDJdb ground truth, filter to Score >= 2, validate and trim CDR3 sequences.
  2. Find matching Mal-ID specimens for that disease from metadata.
  3. For each specimen, find cdr3_aa sequences with >= 90% Levenshtein similarity
     to any ground truth sequence (keeping the best match per sample CDR3).
  4. Collect all matches across all diseases into a single output CSV.

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

DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "vdjdb_matches.csv")

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


def load_ground_truth(disease: str, vdjdb_path: str) -> list[dict]:
    """
    Load VDJdb file, filter Score >= MIN_SCORE, validate CDR3 starts with C
    and ends with F, then strip those terminal residues.

    Returns a list of dicts with keys:
        trimmed_cdr3, cdr3, v, j, score
    """
    df = pd.read_csv(vdjdb_path, sep="\t")

    for col in ("Score", "CDR3", "V", "J"):
        if col not in df.columns:
            sys.exit(f"ERROR: '{col}' column not found in {vdjdb_path}")

    df = df[df["Score"] >= MIN_SCORE].copy()
    tqdm.write(f"[{disease}] {len(df)} sequences after Score >= {MIN_SCORE} filter")

    records = []
    for idx, row in df.iterrows():
        seq = str(row["CDR3"])
        if not seq.startswith("C"):
            tqdm.write(f"  WARNING [{disease}] CDR3 at row {idx} does not start with 'C': {seq!r}")
        if not seq.endswith("F"):
            tqdm.write(f"  WARNING [{disease}] CDR3 at row {idx} does not end with 'F': {seq!r}")
        trimmed = seq
        if trimmed.startswith("C"):
            trimmed = trimmed[1:]
        if trimmed.endswith("F"):
            trimmed = trimmed[:-1]
        records.append({
            "trimmed_cdr3": trimmed,
            "cdr3":  seq,
            "v":     row["V"],
            "j":     row["J"],
            "score": row["Score"],
        })

    tqdm.write(f"[{disease}] {len(records)} ground truth sequences loaded")
    return records


# ---------------------------------------------------------------------------
# Per-disease worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def process_disease(disease: str, vdjdb_path: str, tqdm_position: int) -> list[dict]:
    """
    Process one disease. Returns a list of result row dicts.
    """
    tqdm.write(f"\n[{disease}] Starting (VDJdb: {vdjdb_path})")

    ground_truth = load_ground_truth(disease, vdjdb_path)
    if not ground_truth:
        tqdm.write(f"[{disease}] WARNING: no ground truth sequences, skipped.")
        return []

    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    specimens = (
        metadata[metadata["disease"] == disease][["participant_label", "specimen_label"]]
        .drop_duplicates()
    )
    tqdm.write(f"[{disease}] Found {len(specimens)} specimens")

    if specimens.empty:
        tqdm.write(f"[{disease}] WARNING: no specimens found in metadata.")
        return []

    results = []
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
        filename    = f"part_table_{participant}_{specimen}"

        spec_bar.set_postfix(specimen=specimen)

        if not os.path.exists(tcr_file):
            tqdm.write(f"  WARNING [{disease}] TCR file not found, skipping: {tcr_file}")
            continue

        with gzip.open(tcr_file, "rt") as fh:
            tcr_df = pd.read_csv(fh, sep="\t", low_memory=False)

        for col in ("cdr3_aa", "v_call", "j_call"):
            if col not in tcr_df.columns:
                tqdm.write(f"  WARNING [{disease}] '{col}' column missing in {tcr_file}, skipping.")
                continue

        unique_rows = (
            tcr_df[["cdr3_aa", "v_call", "j_call"]]
            .dropna(subset=["cdr3_aa"])
            .drop_duplicates(subset=["cdr3_aa"])
        )

        n_matched = 0
        for _, sample_row in tqdm(
            unique_rows.iterrows(),
            total=len(unique_rows),
            desc=f"  {specimen} cdr3s",
            unit="seq",
            position=tqdm_position + 1,
            leave=False,
            dynamic_ncols=True,
        ):
            cdr3_str = str(sample_row["cdr3_aa"])

            # Find the best-matching ground truth entry above threshold
            best_sim  = -1.0
            best_gt   = None
            for gt in ground_truth:
                sim = seq_similarity(cdr3_str, gt["trimmed_cdr3"])
                if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    best_gt  = gt

            if best_gt is not None:
                results.append({
                    "disease":            disease,
                    "sample_cdr3":        cdr3_str,
                    "sample_vgene":       sample_row["v_call"],
                    "sample_jgene":       sample_row["j_call"],
                    "public_clone_cdr3":  best_gt["cdr3"],
                    "public_clone_vgene": best_gt["v"],
                    "public_clone_jgene": best_gt["j"],
                    "similarity":         round(best_sim, 6),
                    "score":              best_gt["score"],
                    "filename":           filename,
                })
                n_matched += 1

        tqdm.write(f"  [{disease}] {specimen}: {n_matched} matched sequences")

    tqdm.write(f"[{disease}] Done. {len(results)} total matched rows.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Validate metadata columns up front before spawning workers
    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    required_cols = {"participant_label", "specimen_label", "disease"}
    missing = required_cols - set(metadata.columns)
    if missing:
        sys.exit(f"ERROR: metadata is missing columns: {missing}")

    # Each disease gets its own tqdm row (position 0, 2, 4) with the inner
    # cdr3 bar on the row below (position 1, 3, 5), leaving no overlap.
    jobs = [
        (disease, vdjdb_path, idx * 2)
        for idx, (disease, vdjdb_path) in enumerate(VDJDB_FILES.items())
    ]

    all_results = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(process_disease, disease, vdjdb_path, pos): disease
            for disease, vdjdb_path, pos in jobs
        }
        for future in as_completed(futures):
            disease = futures[future]
            try:
                rows = future.result()
                all_results.extend(rows)
                tqdm.write(f"[{disease}] collected {len(rows)} rows")
            except Exception as exc:
                tqdm.write(f"ERROR [{disease}]: {exc}")

    out_df = pd.DataFrame(all_results, columns=[
        "disease", "sample_cdr3", "sample_vgene", "sample_jgene",
        "public_clone_cdr3", "public_clone_vgene", "public_clone_jgene",
        "similarity", "score", "filename",
    ])
    out_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {output_path}")
    print("All diseases complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match Mal-ID TCR cdr3_aa sequences against VDJdb ground truth."
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write output CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    process(args.output_path)
