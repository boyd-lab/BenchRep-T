# TCR Data Cleanup Script

**Script:** `scripts/clean_tcr_data.py`

## Overview

Reads bz2-compressed TCR internal format files from `data/internal_format/TCR/`, applies light data cleanup, and writes cleaned bz2 TSV files to `data_clean/internal_format_clean/TCR/`. The output preserves all original columns; only the data within is cleaned/filtered/augmented.

## Input / Output

| | Path |
|---|---|
| Input | `data/internal_format/TCR/part_table_<participant>.bz2` |
| Output | `data_clean/internal_format_clean/TCR/part_table_<participant>.bz2` |
| Gene reference | `data/tcrb_v_gene_cdrs.generated.tsv` |
| Reports | `scripts/reports/preprocessing_report_<timestamp>.csv` |
| | `scripts/reports/summary_report_<timestamp>.txt` |
| | `scripts/reports/cleanup_log_<timestamp>.txt` |

## Cleanup Steps (applied in order)

### 1. Remove non-productive sequences
Drop rows where `productive != 't'`.

### 2. Remove low V-score sequences
Drop rows where `v_score <= 80`.

### 3. Clean AA sequences (remove spaces, uppercase)
Strips spaces from the following columns (IgBLAST outputs AA sequences with spaces between each residue, e.g. `" A  S  S  P ..."` → `"ASSPP..."`):
- `cdr3_seq_aa_q`
- `post_seq_aa_q`
- `fr3_seq_aa_q` (also cleaned here even though step 10 will overwrite it, to keep input consistent)

Dots and dashes are **not** removed here — they are left to surface as non-standard characters in step 5, where they are counted and reported before the rows are dropped.

Empty strings after cleaning are converted to NaN.

### 4. Remove sequences with missing critical fields
Drop rows where any of the following are empty or NaN:
- `cdr3_seq_aa_q` (CDR3 amino acid sequence)
- `v_segment` (V gene assignment)
- `j_segment` (J gene assignment)

### 5. Remove sequences with non-standard amino acid characters in CDR3
After space removal, the CDR3 AA should contain only the 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`). Rows with any other character (e.g. `*` for stop codon, `X` for unknown) are dropped.

### 6. Rename indistinguishable V genes
Some V gene names are indistinguishable when using FR3 primers. All alleles of the following genes are renamed to the canonical form with allele `*01`:

| Original (any allele) | Renamed to |
|---|---|
| `TRBV12-4*XX` | `TRBV12-3*01` |
| `TRBV6-3*XX` | `TRBV6-2*01` |

Reference: [Meysman et al., Genome Medicine 2021](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-01008-4)

### 7. Fix specific allele
Exact-match replacement:

| Original | Fixed |
|---|---|
| `TRBV6-2*02` | `TRBV6-2*01` |

Background: IgBLAST can call `TRBV6-2*02`, but this allele has no CDR1/CDR2 annotation available in our reference (it has been renamed). Replacing it with `*01` allows reference lookup to succeed.

### 8. Remove sequences with V gene = TRBV25/OR9-2*01
This gene is a pseudogene and also has no reference in the gene reference file.
It was removed in the original Mal-ID when removing rows with no CDR1/CDR2.

| Gene | Reason |
|---|---|
| `TRBV25/OR9-2*01` | No CDR/FR reference data available |

### 9. Add `sequence_id` column
Constructs `sequence_id` (AIRR convention) by concatenating `run_label + "|" + trimmed_read_id`. This column is added to the output so the cleaned internal format files are directly compatible with AIRR tooling without requiring a separate construction step.

### 10. Always overwrite all FR / CDR1-2 AA columns from reference
Joins `v_segment` against `data/tcrb_v_gene_cdrs.generated.tsv` and **always overwrites** all five FR / CDR1-2 AA columns with the deterministic reference sequence:

- `fr1_seq_aa_q`, `cdr1_seq_aa_q`, `fr2_seq_aa_q`, `cdr2_seq_aa_q`: never populated by IgBLAST for TCR data.
- `fr3_seq_aa_q`: IgBLAST only captures the portion of FR3 that was sequenced (starts inside FR3), not the full gene region.

This matches the behavior of the original Mal-ID code (`etl.py`, `preprocess_each_participant_table`), which explicitly drops all five columns and re-merges from reference.

| Reference column | Internal column |
|---|---|
| `fwr1_aa` | `fr1_seq_aa_q` |
| `cdr1_aa` | `cdr1_seq_aa_q` |
| `fwr2_aa` | `fr2_seq_aa_q` |
| `cdr2_aa` | `cdr2_seq_aa_q` |
| `fwr3_aa` | `fr3_seq_aa_q` |

If a required reference column or a required data column is missing, the script raises an error. If a V gene is not present in the reference table, the affected rows are logged and reported (FR/CDR cells remain as-is for those rows). If a V gene **is** in the reference but has `NaN` for a specific column (partial reference miss), that column is not overwritten for those rows and the discrepancy is logged as a warning.

### 11. Check missing CDR1/CDR2 AA sequences after reference fill
After step 10, counts the number of rows where `cdr1_seq_aa_q` and `cdr2_seq_aa_q` are still NaN. In a healthy run these should be zero (all V genes found in reference). Any remaining missing values correspond to the V genes listed in `step10_reference_miss_genes`.

## Reports

After processing all files, the script writes three output files to `scripts/reports/`:

| File | Contents |
|---|---|
| `preprocessing_report_<timestamp>.csv` | One row per participant. Per-step counts as separate columns. Dict-valued columns (gene fixes, bad chars, reference misses) stored as JSON strings. |
| `summary_report_<timestamp>.txt` | Human-readable aggregated totals across all participants for every step. |
| `cleanup_log_<timestamp>.txt` | Full run log (same content as stdout). |

### CSV columns

| Column | Description |
|---|---|
| `participant_label` | Participant identifier |
| `original_count` | Sequences before any cleanup |
| `step1_non_productive_removed` | Rows dropped (not productive) |
| `step2_low_v_score_removed` | Rows dropped (v_score ≤ 80) |
| `step4_missing_field_counts` | JSON: `{field: rows_with_missing}` before dropping |
| `step4_total_dropped` | Total rows dropped for missing fields |
| `step5_invalid_aa_chars_found` | JSON: `{char: rows_containing_char}` |
| `step5_rows_removed` | Rows dropped for non-standard AA chars |
| `step6_gene_fixes` | JSON: `{"old_allele->new": count}` |
| `step7_allele_fixes` | JSON: `{"old_allele->new": count}` |
| `step8_gene_removed` | JSON: `{v_gene: rows_removed}` — genes removed (not in reference) |
| `step9_sequence_id_added` | True/False |
| `step10_reference_overwritten` | JSON: `{column: rows_overwritten}` — all five FR/CDR columns always overwritten |
| `step10_reference_miss_genes` | JSON: `{v_gene: rows_affected}` — genes not in reference |
| `step10_reference_miss_rows` | Total rows with no reference match |
| `step10_partial_ref_miss` | JSON: `{column: rows_not_overwritten}` — genes in reference but column value is NaN |
| `step11_missing_cdr_after_fill` | JSON: `{column: rows_still_missing}` for cdr1/cdr2 after reference fill |
| `after_clean` | Sequences after all cleanup |
| `total_dropped` | Total sequences removed |

## Output Format Notes

- All original internal format columns are preserved.
- The output is the same bz2-compressed TSV format as the input.
- `sequence_id` is added as a new column.
- `fr1_seq_aa_q`, `cdr1_seq_aa_q`, `fr2_seq_aa_q`, `cdr2_seq_aa_q`, `fr3_seq_aa_q` are always replaced with reference sequences (see step 10).
- `cdr3_seq_aa_q`, `post_seq_aa_q`, and `fr3_seq_aa_q` are space-stripped in place (see step 3).
- All other columns are unchanged.

## Converting to AIRR Format

The cleaned internal format files can be converted to AIRR TSV format using:

```
scripts/clean_tcr_data_to_airr.py
```

See `scripts/CLEAN_TCR_DATA_TO_AIRR.md` for full documentation. When doing so, three things are applied: column renaming, adding empty AIRR-only columns, and optionally fixing boolean value casing.

### 1. Column renaming

Copy-paste ready Python dict (also available as `AIRR_COLUMN_MAPPING` in `clean_tcr_data.py`):

```python
AIRR_COLUMN_MAPPING = {
    "sequence_id":         "sequence_id",          # added by clean_tcr_data.py
    "specimen_label":      "repertoire_id",
    "amplification_locus": "locus",
    "isosubtype":          "c_call",               # BCR only
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

### 2. Empty AIRR-only columns

The following columns are required by the AIRR schema but have no equivalent in the internal format. The existing convert script adds them as empty columns:

```
rev_comp, sequence_alignment, germline_alignment, junction, junction_aa,
v_cigar, d_cigar, j_cigar
```

### 3. Boolean column values: `'t'`/`'f'` vs `'T'`/`'F'`

The internal format represents boolean fields as lowercase strings `'t'` and `'f'`. The existing Mal-ID conversion script (`convert_part_table_to_airr_format.py`) **does not change these values** — the project's AIRR-format files therefore also use `'t'`/`'f'`.

The official [AIRR Community schema](https://docs.airr-community.org/en/stable/datarep/rearrangements.html) specifies boolean fields as `T`/`F` (uppercase). The affected columns are:

| Column (AIRR name) | Internal name |
|---|---|
| `productive` | `productive` |
| `stop_codon` | `stop_codon` |
| `vj_in_frame` | `v_j_in_frame` |

If you need strict AIRR schema compliance (e.g. for use with `airr-tools validate`), convert the values after renaming the columns:

```python
for col in ["productive", "stop_codon", "vj_in_frame"]:
    df[col] = df[col].map({"t": "T", "f": "F"})
```
