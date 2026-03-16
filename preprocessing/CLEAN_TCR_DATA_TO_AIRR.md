# TCR AIRR Format Conversion Script

**Script:** `scripts/clean_tcr_data_to_airr.py`

## Overview

Reads bz2-compressed cleaned internal format files from `data_clean/internal_format_clean/TCR/`, renames columns to AIRR names, adds required empty AIRR-only columns, and writes bz2-compressed AIRR TSV files to `data_clean/airr_format_clean/TCR/`.

**All original columns are preserved.** No data filtering or value transformation is performed. This is a purely structural conversion.

## Input / Output

| | Path |
|---|---|
| Input | `data_clean/internal_format_clean/TCR/part_table_<participant>.bz2` |
| Output | `data_clean/airr_format_clean/TCR/part_table_<participant>.tsv.gz` |
| Log | `scripts/reports/airr_conversion_log_<timestamp>.txt` |

## Prerequisites

Input files must have been produced by `clean_tcr_data.py`. In particular:
- `sequence_id` must already be present (added in step 9 of that script).
- All AA columns must already be space-stripped (step 3) and reference-filled (step 10).

## Conversion Steps (applied in order)

### 1. Column Renaming
Rename the 31 internal format columns that have AIRR equivalents. Columns absent from the file (e.g. `isosubtype` — BCR-only, not present in TCR files) are silently skipped. All other original columns are kept unchanged.

**As a dict (copy-paste ready):**
```python
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
```

### 2. Name Clash Check
After renaming, the script verifies that none of the 8 empty AIRR-only columns (added in step 3) already exist in the DataFrame. Raises `ValueError` if any clash is found. No clashes exist in the current data (verified against TCR files).

### 3. Add Empty AIRR-Only Columns
The following columns are required by the AIRR schema but have no equivalent in the internal format. They are added as empty columns:

```python
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
```

### 4. Column Reordering
Columns are reordered into three groups:
1. **AIRR-renamed columns** — in the order they appear in `AIRR_COLUMN_MAPPING` (i.e., the order written above)
2. **Empty AIRR-only columns** — in `AIRR_EMPTY_COLUMNS` order
3. **Extra internal columns** — in `EXTRA_INTERNAL_COLUMNS` order (see below)

### 5. Column Validation
After reordering, the script verifies that every column in the DataFrame is accounted for in `final_order` and vice versa. Raises `ValueError` if any column is missing or unexpected.

### 6. Write Output
Each file is written as a gzip-compressed TSV named `part_table_<participant>.tsv.gz`. Total columns per output file: **121** (113 original + 8 added).

### 7. Post-Run Checkup
After processing all files, verifies that every expected output file was created. Logs an error and exits with code 1 if any are missing.

## Output Format Notes

- **All 113 original internal format columns are preserved**, with 31 of them renamed to AIRR names.
- 8 empty AIRR-only columns are added.
- Output is gzip-compressed TSV (`.tsv.gz`). Decompress with `gzip.open()` or `gunzip` before passing to strict AIRR schema validators. pandas reads `.tsv.gz` files directly via `pd.read_csv(path, sep="\t")`.
- Boolean fields (`productive`, `stop_codon`, `vj_in_frame`) are stored as `'t'`/`'f'` (lowercase), matching the internal format convention and the original Mal-ID code. However, the official AIRR schema specifies `'T'`/`'F'` (uppercase). 
Convert if strict schema compliance is required:

```python
for col in ["productive", "stop_codon", "vj_in_frame"]:
    df[col] = df[col].map({"t": "T", "f": "F"})
```

## Extra Columns (non-AIRR, kept as-is)

The following 82 columns have no AIRR equivalent and are kept in the output under their original internal format names:

```python
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
```
