#!/usr/bin/env python3
"""
Convert cleaned TCR internal format files to AIRR TSV format.

Reads bz2-compressed cleaned internal format files from
data_clean/internal_format_clean/TCR/, renames columns to AIRR names,
adds required empty AIRR-only columns, and writes bz2-compressed AIRR TSV
files to data_clean/airr_format_clean/TCR/.

All original columns are preserved. No data filtering or value transformation
is performed. This is a purely structural conversion:
  1. Rename the 31 internal format columns to their AIRR equivalents
     (columns absent from the file, e.g. isosubtype which is BCR-only, are
     silently skipped).
  2. Check for name clashes between the empty AIRR-only column names and all
     existing columns after renaming. Raises ValueError if any clash is found.
  3. Add 8 empty AIRR-only columns (rev_comp, sequence_alignment,
     germline_alignment, junction, junction_aa, v_cigar, d_cigar, j_cigar).
  4. Reorder columns: AIRR-renamed columns first (AIRR_COLUMN_MAPPING order),
     then empty AIRR-only columns, then extra internal columns
     (EXTRA_INTERNAL_COLUMNS order).
  5. Validate that the final column set matches expectations (raises ValueError
     if any column is missing or unexpected).
  6. Write as bz2-compressed TSV. Total columns: 113 original + 8 added = 121.
  7. Post-run checkup: verify all expected output files were created.

Prerequisites:
  - Input files must have been produced by clean_tcr_data.py.
    In particular, sequence_id must already be present (added in step 9 of
    that script), and all AA columns must already be clean.

Usage:
    python scripts/clean_tcr_data_to_airr.py

Paths are hardcoded relative to this script's grandparent directory
(data_clean/../ = Mal-ID/).
"""

import bz2
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
MAL_ID_ROOT = SCRIPT_DIR.parent.parent  # data_clean/scripts/ -> data_clean/ -> Mal-ID/

INPUT_DIR = MAL_ID_ROOT / "data_clean" / "internal_format_clean" / "TCR"
OUTPUT_DIR = MAL_ID_ROOT / "data_clean" / "airr_format_clean" / "TCR"
REPORTS_DIR = SCRIPT_DIR / "reports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps internal format column names -> AIRR column names.
# sequence_id is already present in the cleaned files (added by clean_tcr_data.py).
# Columns absent from the file (e.g. isosubtype, BCR-only) are silently skipped.
# All other original columns are kept unchanged alongside the renamed ones.
AIRR_COLUMN_MAPPING = {
    "sequence_id":         "sequence_id",
    "specimen_label":      "repertoire_id",
    "amplification_locus": "locus",
    "isosubtype":          "c_call",               # BCR only — skipped for TCR
    "trimmed_sequence":    "sequence",
    "tcrb_clone_id":       "clone_id",
    "v_segment":           "v_call",
    "d_segment":           "d_call",
    "j_segment":           "j_call",
    "v_score":             "v_score",
    "d_score":             "d_score",
    "j_score":             "j_score",
    "stop_codon":          "stop_codon",
    "v_j_in_frame":        "vj_in_frame",
    "productive":          "productive",
    "fr1_seq_nt_q":        "fwr1",
    "cdr1_seq_nt_q":       "cdr1",
    "fr2_seq_nt_q":        "fwr2",
    "cdr2_seq_nt_q":       "cdr2",
    "fr3_seq_nt_q":        "fwr3",
    "cdr3_seq_nt_q":       "cdr3",
    "post_seq_nt_q":       "fwr4",
    "fr1_seq_aa_q":        "fwr1_aa",
    "cdr1_seq_aa_q":       "cdr1_aa",
    "fr2_seq_aa_q":        "fwr2_aa",
    "cdr2_seq_aa_q":       "cdr2_aa",
    "fr3_seq_aa_q":        "fwr3_aa",
    "cdr3_seq_aa_q":       "cdr3_aa",
    "post_seq_aa_q":       "fwr4_aa",
    "v_sequence":          "v_sequence_alignment",
    "d_sequence":          "d_sequence_alignment",
    "j_sequence":          "j_sequence_alignment",
}

# AIRR-required columns with no equivalent in the internal format.
# Added as empty columns to satisfy schema requirements.
# Verified to have no name clashes with any internal format column (original or renamed).
AIRR_EMPTY_COLUMNS = [
    "rev_comp",
    "sequence_alignment",
    "germline_alignment",
    "junction",
    "junction_aa",
    "v_cigar",
    "d_cigar",
    "j_cigar",
]

# Internal format columns with no AIRR equivalent, kept as-is under their
# original names. Order here defines their order in the output DataFrame.
EXTRA_INTERNAL_COLUMNS = [
    # Participant metadata
    "participant_label", "participant2_label", "participant_alt_label",
    "participant_age", "participant_sex", "participant_ethnicity",
    "participant_diagnosis", "participant_description", "participant_species",
    # Specimen metadata
    "sample_label", "specimen_tissue", "specimen_cell_subset",
    "specimen_time_point", "specimen_collected_on", "specimen_description",
    "specimen_affinity",
    # Amplification metadata
    "amplification_label", "amplification_template", "amplification_done_on",
    "amplification_description", "amplification_type",
    "primer_set", "forward_primer", "reverse_primer",
    # Run / read identifiers
    "part_tcrb_id", "run_id", "run_label", "trimmed_read_id", "replicate_label",
    # Alignment metadata
    "strand",
    "n1_sequence", "n2_sequence", "n1_overlap", "n2_overlap",
    "q_start", "q_end",
    "v_start", "v_end", "d_start", "d_end", "j_start", "j_end",
    # Pre-region NT sequences
    "pre_seq_nt_q", "pre_seq_nt_v", "pre_seq_nt_d", "pre_seq_nt_j",
    # Per-gene NT sequences (V/D/J germline alignments for each region)
    "fr1_seq_nt_v", "fr1_seq_nt_d", "fr1_seq_nt_j",
    "cdr1_seq_nt_v", "cdr1_seq_nt_d", "cdr1_seq_nt_j",
    "fr2_seq_nt_v", "fr2_seq_nt_d", "fr2_seq_nt_j",
    "cdr2_seq_nt_v", "cdr2_seq_nt_d", "cdr2_seq_nt_j",
    "fr3_seq_nt_v", "fr3_seq_nt_d", "fr3_seq_nt_j",
    "cdr3_seq_nt_v", "cdr3_seq_nt_d", "cdr3_seq_nt_j",
    "post_seq_nt_v", "post_seq_nt_d", "post_seq_nt_j",
    # Pre-region AA sequence
    "pre_seq_aa_q",
    # Insertion/deletion counts per region
    "insertions_pre", "insertions_fr1", "insertions_cdr1", "insertions_fr2",
    "insertions_cdr2", "insertions_fr3", "insertions_post",
    "deletions_pre", "deletions_fr1", "deletions_cdr1", "deletions_fr2",
    "deletions_cdr2", "deletions_fr3", "deletions_post",
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_participant_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a cleaned internal format DataFrame to AIRR format.

    All original columns are preserved. Steps:
      1. Rename mapped columns to AIRR names (skip absent ones).
      2. Check for name clashes (raises ValueError if found).
      3. Add empty AIRR-only columns.
      4. Reorder: AIRR-renamed cols, empty AIRR cols, extra internal cols.
      5. Validate final column set (raises ValueError if mismatch).
    """
    # Step 1: rename mapped columns (only those present in the file).
    # Track which AIRR output names will be present, in mapping order.
    airr_cols_present = [v for k, v in AIRR_COLUMN_MAPPING.items() if k in df.columns]
    rename_map = {k: v for k, v in AIRR_COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Step 2: check for name clashes between empty AIRR columns and existing columns
    existing = set(df.columns)
    clashes = [col for col in AIRR_EMPTY_COLUMNS if col in existing]
    if clashes:
        raise ValueError(
            f"Name clash: AIRR empty column(s) {clashes} already exist in the "
            f"DataFrame after renaming. Check AIRR_COLUMN_MAPPING and AIRR_EMPTY_COLUMNS."
        )

    # Step 3: add empty AIRR-only columns
    for col in AIRR_EMPTY_COLUMNS:
        df[col] = pd.Series(dtype="object")

    # Step 4: reorder columns
    extra_cols_present = [c for c in EXTRA_INTERNAL_COLUMNS if c in df.columns]
    final_order = airr_cols_present + AIRR_EMPTY_COLUMNS + extra_cols_present

    # Step 5: validate — every column must be accounted for, no extras missed
    expected = set(final_order)
    actual = set(df.columns)
    if expected != actual:
        missing = expected - actual
        unexpected = actual - expected
        raise ValueError(
            f"Column mismatch after reordering. "
            f"Missing from final_order: {missing}. "
            f"In DataFrame but not in final_order: {unexpected}."
        )

    return df[final_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / f"airr_conversion_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("TCR INTERNAL FORMAT -> AIRR FORMAT CONVERSION")
    logger.info(f"Started: {timestamp}")
    logger.info("=" * 60)
    logger.info(f"Input:   {INPUT_DIR}")
    logger.info(f"Output:  {OUTPUT_DIR}")
    logger.info(f"Reports: {REPORTS_DIR}")

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_DIR.glob("part_table_*.bz2"))
    logger.info(f"\nFound {len(input_files)} input files\n")

    n_ok = 0
    n_err = 0

    for idx, input_path in enumerate(input_files, 1):
        participant_label = input_path.stem.replace("part_table_", "")
        output_path = OUTPUT_DIR / input_path.name

        if idx % 50 == 0 or idx == 1:
            logger.info(f"[{idx}/{len(input_files)}] {participant_label}")

        try:
            with bz2.open(input_path, "rt") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)

            df_airr = convert_participant_df(df)

            with bz2.open(output_path, "wt") as f:
                df_airr.to_csv(f, sep="\t", index=False)

            logger.info(f"  {len(df_airr):,} sequences -> {output_path.name}")
            n_ok += 1

        except Exception as e:
            logger.error(f"  ERROR processing {input_path.name}: {e}", exc_info=True)
            n_err += 1

    # ------------------------------------------------------------------
    # Post-run checkup: verify all expected output files were created
    # ------------------------------------------------------------------
    logger.info("\nPost-run checkup: verifying output files...")
    missing_outputs = [
        f.name for f in input_files
        if not (OUTPUT_DIR / f.name).exists()
    ]
    if missing_outputs:
        logger.error(f"  MISSING {len(missing_outputs)}/{len(input_files)} output files:")
        for name in missing_outputs:
            logger.error(f"    {name}")
    else:
        logger.info(f"  OK: all {len(input_files)} output files present.")

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE: {n_ok} files converted, {n_err} errors")
    logger.info(f"Output:  {OUTPUT_DIR}")
    logger.info(f"Reports: {REPORTS_DIR}")
    logger.info("=" * 60)

    return 0 if (n_err == 0 and not missing_outputs) else 1


if __name__ == "__main__":
    sys.exit(main())
