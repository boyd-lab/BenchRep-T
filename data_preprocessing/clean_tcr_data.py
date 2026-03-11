#!/usr/bin/env python3
"""
Light data cleanup for TCR internal format files.

Reads bz2-compressed internal format TSV files from data/internal_format/TCR/,
applies the following cleanup steps, and writes cleaned bz2 TSV files to
data_clean/internal_format_clean/TCR/.

Cleanup steps (in order):
  1. Remove non-productive sequences (productive != 't')
  2. Remove sequences with v_score <= 80
  3. Clean AA sequences: remove spaces from cdr3_seq_aa_q, post_seq_aa_q, and fr3_seq_aa_q
  4. Remove sequences with empty/NaN CDR3 AA, V gene (v_segment), or J gene (j_segment)
  5. Remove sequences containing non-standard AA chars in CDR3 (e.g. *, X)
  6. Rename indistinguishable V genes (prefix match, all alleles replaced with *01):
       TRBV12-4 -> TRBV12-3
       TRBV6-3  -> TRBV6-2
  7. Fix specific allele: TRBV6-2*02 -> TRBV6-2*01
  8. Remove sequences with V gene not in reference (TRBV25/OR9-2*01)
  9. Add sequence_id column: run_label + "|" + trimmed_read_id
 10. Overwrite fr1/cdr1/fr2/cdr2/fr3_seq_aa_q from reference (always, for all columns).
     FR3 from IgBLAST is partial (sequencing starts inside FR3); FR1/CDR1/FR2/CDR2 are never
     populated by IgBLAST for TCR. Matches original Mal-ID etl.py behavior.

Output format: same internal format columns as input, saved as bz2-compressed TSV.
AIRR format users can convert using the AIRR_COLUMN_MAPPING dict in this file,
which mirrors data/Maxim-malid-release-202408/scripts/convert_part_table_to_airr_format.py.

Usage:
    python scripts/clean_tcr_data.py

Paths are hardcoded relative to this script's grandparent directory
(data_clean/../ = Mal-ID/).
"""

import bz2
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
MAL_ID_ROOT = SCRIPT_DIR.parent.parent  # data_clean/scripts/ -> data_clean/ -> Mal-ID/

INPUT_DIR = MAL_ID_ROOT / "data" / "internal_format" / "TCR"
OUTPUT_DIR = MAL_ID_ROOT / "data_clean" / "internal_format_clean" / "TCR"
GENE_REFERENCE_PATH = MAL_ID_ROOT / "data" / "tcrb_v_gene_cdrs.generated.tsv"
REPORTS_DIR = SCRIPT_DIR / "reports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V_SCORE_THRESHOLD = 80  # strictly greater than

# Prefix-match gene fixes: any allele of old gene -> new gene + *01
# [From original malid repo: Replace indistinguishable TRBV gene names [due to FR3 primers] with the version that we use in our data.
# https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4 ]
GENE_FIXES = {
    "TRBV12-4": "TRBV12-3",
    "TRBV6-3": "TRBV6-2",
}


# Exact-match allele fixes
# [From original malid repo: Our old IgBLAST can generate TRBV6-2*02 calls, but no CDR1+2 information is available for this allele from get_tcr_v_gene_annotations, because it has been renamed:
# https://www.imgt.org/IMGTrepertoire/index.php?section=LocusGenes&repertoire=genetable&species=human&group=TRBV - see (40) ]
GENE_ALLELE_FIXES = {
    "TRBV6-2*02": "TRBV6-2*01",
}

# V gene names to remove entirely (no reference data available; cannot be used in downstream analysis)
GENES_TO_REMOVE = {"TRBV25/OR9-2*01"}

# Standard 20 amino acids
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Columns in the reference file that map to internal format CDR/FR columns
REFERENCE_COL_MAP = {
    "fwr1_aa": "fr1_seq_aa_q",
    "cdr1_aa": "cdr1_seq_aa_q",
    "fwr2_aa": "fr2_seq_aa_q",
    "cdr2_aa": "cdr2_seq_aa_q",
    "fwr3_aa": "fr3_seq_aa_q",
}

# Column mapping for AIRR format conversion (not use in this script. Only for reference / downstream users)
# internal_format_col -> airr_col
AIRR_COLUMN_MAPPING = {
    "sequence_id": "sequence_id",          # added by this script
    "specimen_label": "repertoire_id",
    "amplification_locus": "locus",
    "isosubtype": "c_call",                # BCR only
    "trimmed_sequence": "sequence",
    "tcrb_clone_id": "clone_id",
    "v_segment": "v_call",
    "d_segment": "d_call",
    "j_segment": "j_call",
    "v_score": "v_score",
    "d_score": "d_score",
    "j_score": "j_score",
    "stop_codon": "stop_codon",
    "v_j_in_frame": "vj_in_frame",
    "productive": "productive",
    "fr1_seq_nt_q": "fwr1",
    "cdr1_seq_nt_q": "cdr1",
    "fr2_seq_nt_q": "fwr2",
    "cdr2_seq_nt_q": "cdr2",
    "fr3_seq_nt_q": "fwr3",
    "cdr3_seq_nt_q": "cdr3",
    "post_seq_nt_q": "fwr4",
    "fr1_seq_aa_q": "fwr1_aa",
    "cdr1_seq_aa_q": "cdr1_aa",
    "fr2_seq_aa_q": "fwr2_aa",
    "cdr2_seq_aa_q": "cdr2_aa",
    "fr3_seq_aa_q": "fwr3_aa",
    "cdr3_seq_aa_q": "cdr3_aa",
    "post_seq_aa_q": "fwr4_aa",
    "v_sequence": "v_sequence_alignment",
    "d_sequence": "d_sequence_alignment",
    "j_sequence": "j_sequence_alignment",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gene_reference(path: Path) -> pd.DataFrame:
    """Load tcrb_v_gene_cdrs.generated.tsv and return indexed on v_call."""
    ref = pd.read_csv(path, sep="\t")
    expected_cols = ["v_call"] + list(REFERENCE_COL_MAP.keys())
    missing = [c for c in expected_cols if c not in ref.columns]
    if missing:
        raise ValueError(f"Gene reference file is missing expected columns: {missing}")
    ref = ref[expected_cols].copy()
    ref = ref.drop_duplicates(subset="v_call")
    ref = ref.set_index("v_call")
    return ref


def clean_aa_sequence(series: pd.Series) -> pd.Series:
    """
    Strip spaces from AA sequence column (IgBLAST outputs AA with spaces between residues).
    Returns uppercase string; empty strings become NaN.
    """
    cleaned = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )
    return cleaned.replace({"": np.nan, "NAN": np.nan})


def find_invalid_aa_chars(sequence: str) -> set:
    """Return the set of non-standard AA characters found in a sequence."""
    if pd.isna(sequence) or sequence == "":
        return set()
    return set(str(sequence).upper()) - VALID_AMINO_ACIDS


def clean_participant_df(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    participant_label: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply all cleanup steps to a participant DataFrame.

    Returns:
        (cleaned_df, stats_dict) where stats_dict has per-step details suitable
        for the preprocessing report CSV.
    """
    stats: Dict = {"participant_label": participant_label, "original_count": len(df)}

    # ------------------------------------------------------------------
    # Step 1: Remove non-productive
    # ------------------------------------------------------------------
    if "productive" not in df.columns:
        raise ValueError("Required column 'productive' not found")
    mask = df["productive"] == "t"
    stats["step1_non_productive_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # ------------------------------------------------------------------
    # Step 2: Remove v_score <= threshold
    # ------------------------------------------------------------------
    if "v_score" not in df.columns:
        raise ValueError("Required column 'v_score' not found")
    mask = df["v_score"] > V_SCORE_THRESHOLD
    stats["step2_low_v_score_removed"] = int((~mask).sum())
    df = df[mask].copy()

    # ------------------------------------------------------------------
    # Step 3: Clean AA sequences (remove spaces, uppercase)
    # IgBLAST outputs AA sequences with spaces between each residue.
    # fr3_seq_aa_q is also cleaned here even though step 10 will overwrite it,
    # to keep the input to step 10 consistent.
    # Dots and dashes are left as-is — they will surface as non-standard
    # characters in step 5 and be reported there.
    # ------------------------------------------------------------------
    for col in ["cdr3_seq_aa_q", "post_seq_aa_q", "fr3_seq_aa_q"]:
        if col in df.columns:
            df[col] = clean_aa_sequence(df[col])

    # ------------------------------------------------------------------
    # Step 4: Remove sequences with empty/NaN CDR3, V gene, or J gene
    # ------------------------------------------------------------------
    required_cols = [c for c in ["cdr3_seq_aa_q", "v_segment", "j_segment"] if c in df.columns]
    step4_per_field = {}
    for col in required_cols:
        n_missing = int((df[col].isna() | (df[col] == "")).sum())
        step4_per_field[col] = n_missing

    before = len(df)
    for col in required_cols:
        df = df[df[col].notna() & (df[col] != "")].copy()
    stats["step4_missing_field_counts"] = step4_per_field
    stats["step4_total_dropped"] = before - len(df)

    # ------------------------------------------------------------------
    # Step 5: Remove sequences with non-standard AA chars in CDR3
    # ------------------------------------------------------------------
    if "cdr3_seq_aa_q" not in df.columns:
        raise ValueError("Required column 'cdr3_seq_aa_q' not found")
    # Per-character counts: char -> number of rows containing that char
    char_row_counts: Dict[str, int] = defaultdict(int)
    bad_rows = []
    for idx, seq in df["cdr3_seq_aa_q"].items():
        bad_chars = find_invalid_aa_chars(seq)
        if bad_chars:
            bad_rows.append(idx)
            for ch in bad_chars:
                char_row_counts[ch] += 1

    stats["step5_invalid_aa_chars_found"] = dict(char_row_counts)
    stats["step5_rows_removed"] = len(bad_rows)
    df = df.drop(index=bad_rows).copy()

    # ------------------------------------------------------------------
    # Step 6: Rename indistinguishable V genes (gene prefix match (any allele) -> new_gene*01)
    # ------------------------------------------------------------------
    if "v_segment" not in df.columns:
        raise ValueError("Required column 'v_segment' not found")
    step6_fixes: Dict[str, int] = {}
    for old_gene, new_gene in GENE_FIXES.items():
        prefix = f"{old_gene}*"
        mask = df["v_segment"].str.startswith(prefix, na=False)
        if mask.any():
            # Record each unique old allele separately
            for old_allele, count in df.loc[mask, "v_segment"].value_counts().items():
                new_name = f"{new_gene}*01"
                step6_fixes[f"{old_allele}->{new_name}"] = int(count)
            df.loc[mask, "v_segment"] = f"{new_gene}*01"
    stats["step6_gene_fixes"] = step6_fixes

    # ------------------------------------------------------------------
    # Step 7: Fix specific alleles (exact match)
    # ------------------------------------------------------------------
    step7_fixes: Dict[str, int] = {}
    for old_allele, new_allele in GENE_ALLELE_FIXES.items():
        mask = df["v_segment"] == old_allele
        if mask.any():
            step7_fixes[f"{old_allele}->{new_allele}"] = int(mask.sum())
            df.loc[mask, "v_segment"] = new_allele
    stats["step7_allele_fixes"] = step7_fixes

    # ------------------------------------------------------------------
    # Step 8: Remove sequences with V gene = TRBV25/OR9-2*01
    # This gene is a pseudogene and also has no reference in the gene reference file.
    # It was removed in the original Mal-ID when removing rows with no CDR1/CDR2.
    # ------------------------------------------------------------------
    step8_gene_removed: Dict[str, int] = {}
    mask = df["v_segment"].isin(GENES_TO_REMOVE)
    if mask.any():
        for gene, count in df.loc[mask, "v_segment"].value_counts().items():
            step8_gene_removed[gene] = int(count)
        df = df[~mask].copy()
    stats["step8_gene_removed"] = step8_gene_removed

    # ------------------------------------------------------------------
    # Step 9: Add sequence_id column
    # ------------------------------------------------------------------
    for req in ["run_label", "trimmed_read_id"]:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' not found")
    df["sequence_id"] = df["run_label"] + "|" + df["trimmed_read_id"].astype(str)
    stats["step9_sequence_id_added"] = True

    # ------------------------------------------------------------------
    # Step 10: Always overwrite fr1/cdr1/fr2/cdr2/fr3_seq_aa_q from reference.
    #
    # All five FR/CDR AA columns are replaced with the deterministic reference
    # sequence for the called V gene. FR3 from IgBLAST is partial (sequencing
    # starts inside FR3); FR1/CDR1/FR2/CDR2 are never populated by IgBLAST for
    # TCR data. Matches original Mal-ID etl.py behavior.
    # ------------------------------------------------------------------
    step10_overwritten: Dict[str, int] = {}
    step10_miss_genes: Dict[str, int] = {}   # v_segment value -> row count, not in reference
    step10_partial_ref_miss: Dict[str, int] = {}  # internal_col -> rows in ref but col value is NaN

    # Detect genes not in reference (after all fixes applied)
    not_in_ref = ~df["v_segment"].isin(ref.index)
    if not_in_ref.any():
        for gene, count in df.loc[not_in_ref, "v_segment"].value_counts().items():
            step10_miss_genes[gene] = int(count)

    in_ref_mask = df["v_segment"].isin(ref.index)

    for ref_col, internal_col in REFERENCE_COL_MAP.items():
        if ref_col not in ref.columns:
            raise ValueError(f"Gene reference file is missing required column '{ref_col}'")
        if internal_col not in df.columns:
            raise ValueError(f"Required column '{internal_col}' not found in data")

        ref_values = df["v_segment"].map(ref[ref_col])
        overwrite_mask = ref_values.notna()

        # Detect genes that ARE in the reference but have NaN for this specific column
        partial_miss_mask = in_ref_mask & ref_values.isna()
        if partial_miss_mask.any():
            step10_partial_ref_miss[internal_col] = int(partial_miss_mask.sum())

        if overwrite_mask.any():
            if df[internal_col].dtype != object:
                df[internal_col] = df[internal_col].astype(object)
            df.loc[overwrite_mask, internal_col] = ref_values[overwrite_mask].values
            step10_overwritten[internal_col] = int(overwrite_mask.sum())

    stats["step10_reference_overwritten"] = step10_overwritten
    stats["step10_reference_miss_genes"] = step10_miss_genes
    stats["step10_reference_miss_rows"] = sum(step10_miss_genes.values())
    stats["step10_partial_ref_miss"] = step10_partial_ref_miss

    # ------------------------------------------------------------------
    # Step 11: Report missing CDR1/CDR2 AA sequences after reference fill
    # These should be zero if the reference covers all V genes. Any missing
    # values indicate V genes not found in the reference (see step10_reference_miss_genes).
    # ------------------------------------------------------------------
    step11_missing: Dict[str, int] = {}
    for col in ["cdr1_seq_aa_q", "cdr2_seq_aa_q"]:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            step11_missing[col] = n_missing
    stats["step11_missing_cdr_after_fill"] = step11_missing

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats["after_clean"] = len(df)
    stats["total_dropped"] = stats["original_count"] - len(df)

    return df, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report_row(stats: Dict) -> Dict:
    """Flatten a stats dict into a single CSV-friendly row."""
    row = {
        "participant_label": stats["participant_label"],
        "original_count": stats["original_count"],
        "step1_non_productive_removed": stats.get("step1_non_productive_removed", 0),
        "step2_low_v_score_removed": stats.get("step2_low_v_score_removed", 0),
        "step4_missing_field_counts": json.dumps(stats.get("step4_missing_field_counts", {})),
        "step4_total_dropped": stats.get("step4_total_dropped", 0),
        "step5_invalid_aa_chars_found": json.dumps(stats.get("step5_invalid_aa_chars_found", {})),
        "step5_rows_removed": stats.get("step5_rows_removed", 0),
        "step6_gene_fixes": json.dumps(stats.get("step6_gene_fixes", {})),
        "step7_allele_fixes": json.dumps(stats.get("step7_allele_fixes", {})),
        "step8_gene_removed": json.dumps(stats.get("step8_gene_removed", {})),
        "step9_sequence_id_added": stats.get("step9_sequence_id_added", False),
        "step10_reference_overwritten": json.dumps(stats.get("step10_reference_overwritten", {})),
        "step10_reference_miss_genes": json.dumps(stats.get("step10_reference_miss_genes", {})),
        "step10_reference_miss_rows": stats.get("step10_reference_miss_rows", 0),
        "step10_partial_ref_miss": json.dumps(stats.get("step10_partial_ref_miss", {})),
        "step11_missing_cdr_after_fill": json.dumps(stats.get("step11_missing_cdr_after_fill", {})),
        "after_clean": stats.get("after_clean", 0),
        "total_dropped": stats.get("total_dropped", 0),
    }
    return row


def write_csv_report(all_stats: List[Dict], path: Path):
    """Write per-participant preprocessing report CSV."""
    rows = [build_report_row(s) for s in all_stats]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_text_report(all_stats: List[Dict], path: Path, n_errors: int, n_empty: int,
                      ref_gene_count: int, timestamp: str):
    """Write human-readable summary text report."""

    total_participants = len(all_stats)
    total_original = sum(s["original_count"] for s in all_stats)
    total_after = sum(s["after_clean"] for s in all_stats)
    total_dropped = total_original - total_after

    # Aggregate per-step totals
    total_step1 = sum(s.get("step1_non_productive_removed", 0) for s in all_stats)
    total_step2 = sum(s.get("step2_low_v_score_removed", 0) for s in all_stats)
    total_step4 = sum(s.get("step4_total_dropped", 0) for s in all_stats)
    total_step5 = sum(s.get("step5_rows_removed", 0) for s in all_stats)

    # Step 4: per-field missing counts
    step4_field_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for field, count in s.get("step4_missing_field_counts", {}).items():
            step4_field_totals[field] += count

    # Step 5: aggregate bad char counts
    step5_char_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for ch, count in s.get("step5_invalid_aa_chars_found", {}).items():
            step5_char_totals[ch] += count

    # Steps 6-7: aggregate all gene/allele fixes
    step6_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for fix, count in s.get("step6_gene_fixes", {}).items():
            step6_totals[fix] += count

    step7_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for fix, count in s.get("step7_allele_fixes", {}).items():
            step7_totals[fix] += count

    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    def sub(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    h("TCR DATA CLEANUP — PREPROCESSING REPORT")
    lines.append(f"Generated : {timestamp}")
    lines.append(f"Input dir : {INPUT_DIR}")
    lines.append(f"Output dir: {OUTPUT_DIR}")
    lines.append(f"Reference : {GENE_REFERENCE_PATH}  ({ref_gene_count} V gene entries)")

    h("OVERALL SUMMARY")
    lines.append(f"Participants processed : {total_participants}")
    lines.append(f"Files with errors      : {n_errors}")
    lines.append(f"Files empty/all-removed: {n_empty}")
    lines.append(f"Total sequences (input): {total_original:,}")
    lines.append(f"Total sequences (output): {total_after:,}")
    lines.append(f"Total sequences dropped : {total_dropped:,}  ({total_dropped/total_original*100:.1f}%)" if total_original else "Total sequences dropped : 0")

    h("PER-STEP BREAKDOWN")

    sub("Step 1 — Remove non-productive sequences (productive != 't')")
    lines.append(f"  Rows removed: {total_step1:,}")

    sub("Step 2 — Remove low V-score sequences (v_score <= 80)")
    lines.append(f"  Rows removed: {total_step2:,}")

    sub("Step 3 — Clean AA sequences (remove spaces, uppercase)")
    lines.append("  Spaces stripped from cdr3_seq_aa_q, post_seq_aa_q, and fr3_seq_aa_q (IgBLAST inserts spaces between residues).")
    lines.append("  Dots/dashes are NOT removed here — they surface as non-standard characters in step 5.")

    sub("Step 4 — Remove sequences with empty/NaN CDR3, V gene, or J gene")
    lines.append("  Missing values per field (before dropping):")
    for field, count in sorted(step4_field_totals.items()):
        lines.append(f"    {field}: {count:,} rows with missing value")
    if not step4_field_totals:
        lines.append("    (none)")
    lines.append(f"  Total rows dropped: {total_step4:,}")

    sub("Step 5 — Remove sequences with non-standard AA characters in CDR3")
    lines.append("  Non-standard characters found (rows containing each char):")
    if step5_char_totals:
        for ch, count in sorted(step5_char_totals.items(), key=lambda x: -x[1]):
            label = {
                "*": "'*' (stop codon)",
                "X": "'X' (unknown residue)",
                ".": "'.' (gap)",
                "-": "'-' (gap/deletion)",
            }.get(ch, f"'{ch}'")
            lines.append(f"    {label}: {count:,} rows")
    else:
        lines.append("    None found.")
    lines.append(f"  Total rows removed: {total_step5:,}")

    sub("Step 6 — Rename indistinguishable V genes (prefix match -> *01)")
    if step6_totals:
        for fix, count in sorted(step6_totals.items(), key=lambda x: -x[1]):
            lines.append(f"    {fix}: {count:,} rows")
    else:
        lines.append("    No fixes applied.")

    sub("Step 7 — Fix specific alleles (exact match)")
    if step7_totals:
        for fix, count in sorted(step7_totals.items(), key=lambda x: -x[1]):
            lines.append(f"    {fix}: {count:,} rows")
    else:
        lines.append("    No fixes applied.")

    # Step 8: aggregate removed unknown-gene rows
    step8_gene_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for gene, count in s.get("step8_gene_removed", {}).items():
            step8_gene_totals[gene] += count
    total_step8 = sum(step8_gene_totals.values())

    sub("Step 8 — Remove sequences with V gene = TRBV25/OR9-2*01 (pseudogene, no reference data)")
    if step8_gene_totals:
        for gene, count in sorted(step8_gene_totals.items(), key=lambda x: -x[1]):
            lines.append(f"    {gene}: {count:,} rows removed")
        lines.append(f"  Total rows removed: {total_step8:,}")
    else:
        lines.append("    No sequences removed.")

    sub("Step 9 — Add sequence_id column (run_label + '|' + trimmed_read_id)")
    n_id_added = sum(1 for s in all_stats if s.get("step9_sequence_id_added"))
    lines.append(f"  sequence_id added in {n_id_added}/{total_participants} participant files.")

    # Step 10: aggregate overwritten counts and partial ref misses
    step10_overwritten_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for col, count in s.get("step10_reference_overwritten", {}).items():
            step10_overwritten_totals[col] += count

    step10_miss_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for gene, count in s.get("step10_reference_miss_genes", {}).items():
            step10_miss_totals[gene] += count
    total_step10_miss_rows = sum(s.get("step10_reference_miss_rows", 0) for s in all_stats)

    step10_partial_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for col, count in s.get("step10_partial_ref_miss", {}).items():
            step10_partial_totals[col] += count

    sub("Step 10 — Overwrite all FR/CDR AA columns from reference (fr1/cdr1/fr2/cdr2/fr3)")
    lines.append("  All five columns always overwritten with reference sequence per V gene.")
    lines.append("  Rows overwritten per column:")
    for col, count in sorted(step10_overwritten_totals.items()):
        lines.append(f"    {col}: {count:,} rows")
    if not step10_overwritten_totals:
        lines.append("    (none)")
    if step10_miss_totals:
        lines.append(f"  V genes not found in reference ({total_step10_miss_rows:,} rows affected):")
        for gene, count in sorted(step10_miss_totals.items(), key=lambda x: -x[1]):
            lines.append(f"    {gene}: {count:,} rows — FR/CDR NOT filled")
    else:
        lines.append("  All V genes found in reference. No missing FR/CDR fills.")
    if step10_partial_totals:
        lines.append("  V genes in reference but missing value for specific column (partial reference miss):")
        for col, count in sorted(step10_partial_totals.items()):
            lines.append(f"    {col}: {count:,} rows NOT overwritten (reference value is NaN for this column)")
    else:
        lines.append("  No partial reference misses (all reference entries have values for all columns).")

    # Step 11: aggregate missing CDR1/CDR2 after fill
    step11_totals: Dict[str, int] = defaultdict(int)
    for s in all_stats:
        for col, count in s.get("step11_missing_cdr_after_fill", {}).items():
            step11_totals[col] += count

    sub("Step 11 — Check missing CDR1/CDR2 AA sequences after reference fill")
    lines.append("  Missing values indicate V genes not found in the reference (see step 10 reference misses).")
    any_missing = False
    for col, count in sorted(step11_totals.items()):
        if count > 0:
            lines.append(f"    {col}: {count:,} rows still missing after fill")
            any_missing = True
    if not any_missing:
        lines.append("    All CDR1/CDR2 sequences filled. No missing values.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / f"cleanup_log_{timestamp}.txt"

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
    logger.info("TCR INTERNAL FORMAT DATA CLEANUP")
    logger.info(f"Started: {timestamp}")
    logger.info("=" * 60)
    logger.info(f"Input:     {INPUT_DIR}")
    logger.info(f"Output:    {OUTPUT_DIR}")
    logger.info(f"Reference: {GENE_REFERENCE_PATH}")
    logger.info(f"Reports:   {REPORTS_DIR}")

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return 1
    if not GENE_REFERENCE_PATH.exists():
        logger.error(f"Gene reference file not found: {GENE_REFERENCE_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load gene reference
    logger.info("\nLoading gene reference...")
    ref = load_gene_reference(GENE_REFERENCE_PATH)
    logger.info(f"  Loaded {len(ref)} V gene entries")

    # Collect input files
    input_files = sorted(INPUT_DIR.glob("part_table_*.bz2"))
    logger.info(f"\nFound {len(input_files)} input files\n")

    n_ok = 0
    n_err = 0
    n_empty = 0
    all_stats: List[Dict] = []

    for idx, input_path in enumerate(input_files, 1):
        participant_label = input_path.stem.replace("part_table_", "")
        output_path = OUTPUT_DIR / input_path.name

        if idx % 50 == 0 or idx == 1:
            logger.info(f"[{idx}/{len(input_files)}] {participant_label}")

        try:
            with bz2.open(input_path, "rt") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)

            if df.empty:
                logger.warning(f"  Empty file: {input_path.name}")
                n_empty += 1
                continue

            df_clean, stats = clean_participant_df(df, ref, participant_label)
            all_stats.append(stats)

            logger.info(
                f"  {stats['original_count']:,} -> {stats['after_clean']:,} sequences "
                f"(dropped {stats['total_dropped']:,}: "
                f"non_productive={stats['step1_non_productive_removed']}, "
                f"low_v_score={stats['step2_low_v_score_removed']}, "
                f"missing_fields={stats['step4_total_dropped']}, "
                f"invalid_aa={stats['step5_rows_removed']}, "
                f"unknown_gene={sum(stats['step8_gene_removed'].values())})"
            )
            if stats["step6_gene_fixes"]:
                logger.info(f"  gene fixes: {stats['step6_gene_fixes']}")
            if stats["step7_allele_fixes"]:
                logger.info(f"  allele fixes: {stats['step7_allele_fixes']}")
            if stats["step8_gene_removed"]:
                logger.info(f"  unknown gene removed: {stats['step8_gene_removed']}")
            if stats["step10_partial_ref_miss"]:
                logger.warning(
                    f"  partial reference miss (gene in ref but column NaN): "
                    f"{stats['step10_partial_ref_miss']}"
                )

            if df_clean.empty:
                logger.warning(f"  All sequences removed for {participant_label}")
                n_empty += 1
                continue

            with bz2.open(output_path, "wt") as f:
                df_clean.to_csv(f, sep="\t", index=False)

            n_ok += 1

        except Exception as e:
            logger.error(f"  ERROR processing {input_path.name}: {e}", exc_info=True)
            n_err += 1

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    logger.info("\nGenerating reports...")

    csv_path = REPORTS_DIR / f"preprocessing_report_{timestamp}.csv"
    write_csv_report(all_stats, csv_path)
    logger.info(f"  CSV report : {csv_path.name}")

    txt_path = REPORTS_DIR / f"summary_report_{timestamp}.txt"
    write_text_report(all_stats, txt_path, n_errors=n_err, n_empty=n_empty,
                      ref_gene_count=len(ref), timestamp=timestamp)
    logger.info(f"  Text report: {txt_path.name}")

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE: {n_ok} files cleaned, {n_empty} empty/all-removed, {n_err} errors")
    logger.info(f"Output:  {OUTPUT_DIR}")
    logger.info(f"Reports: {REPORTS_DIR}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Post-run checkup: verify all expected output files were created
    # ------------------------------------------------------------------
    logger.info("\nPost-run checkup: verifying output files...")
    missing_outputs = []
    for input_path in input_files:
        expected_output = OUTPUT_DIR / input_path.name
        if not expected_output.exists():
            missing_outputs.append(input_path.name)

    if missing_outputs:
        logger.error(f"  MISSING {len(missing_outputs)}/{len(input_files)} output files:")
        for name in missing_outputs:
            logger.error(f"    {name}")
    else:
        logger.info(f"  OK: all {len(input_files)} output files present.")

    return 0 if (n_err == 0 and not missing_outputs) else 1


if __name__ == "__main__":
    sys.exit(main())
