# AIRR Split-by-Specimen Script

**Script:** `scripts/clean_airr_split_by_specimen.py`

## Overview

Reads every gzip-compressed TCR AIRR TSV file from `airr_format_clean/TCR/` (one file per participant, potentially containing multiple specimens), splits rows by the `repertoire_id` column (=specimen name), and writes one output file per specimen — **but only for specimens that appear in `metadata.tsv`**.

Specimens present in the AIRR data but absent from the metadata are skipped and reported. This ensures the output directory contains exactly the specimens that are meaningful for downstream analysis (i.e., those with a known disease label, cross-validation fold assignment, and other metadata).

## Why specimens are skipped

The AIRR data contains **703 unique `repertoire_id` values** across 542 participant files, but the metadata only covers **616 specimens**. Of those 616, only **550** have matching TCR data files (51 participants — 66 specimens — exist in the metadata but have only BCR data). The remaining **153 `repertoire_id` values** exist in the AIRR data but have no metadata entry: they are time-point specimens from longitudinal studies (M369, M371, M418, M433, etc.) that were not included in the metadata.

| Population | Count |
|---|---|
| Unique `repertoire_id` (=specimen name) values in TCR AIRR data | 703 |
| Unique `specimen_label` values in metadata | 616 |
| Written to output (in both TCR AIRR data and metadata) | 550 |
| Skipped — in TCR AIRR data but not in metadata | 153 |
| Missing — in metadata but no TCR AIRR file | 66 |

## Input / Output

| | Path |
|---|---|
| Input (AIRR participant files) | `data_clean/airr_format_clean/TCR/part_table_<participant>.tsv.gz` |
| Metadata | `data_clean/metadata.tsv` |
| Output (specimen files) | `data_clean/airr_format_clean_by_specimen/TCR/part_table_<participant>_<specimen>.tsv.gz` |
| Reports directory | `scripts/reports/clean_airr_split_by_specimen/` |

## Output filename convention

```
part_table_<participant_label>_<repertoire_id>.tsv.gz
```

Example: `part_table_BFI-0009950_M433-S001.tsv.gz`

## Reports

All report files are written to `scripts/reports/clean_airr_split_by_specimen/`:

| File | Contents |
|---|---|
| `specimen_overview.csv` | One row per specimen found in the AIRR data. Columns: `participant_label`, `specimen_label`, `in_metadata`, `output_filename`, then all metadata columns joined from `metadata.tsv`. Specimens not in metadata have NaN for metadata columns. |
| `split_by_specimen_report_<timestamp>.txt` | Human-readable summary: counts, disease distribution of written specimens, list of all skipped specimens, and post-run validation result. |
| `split_by_specimen_log_<timestamp>.txt` | Full run log (same content as stdout). |

## Logic

1. Load `metadata.tsv` and build the set of known `specimen_label` values.
2. For each `part_table_<participant>.tsv.gz` in `airr_format_clean/TCR/`:
   - Raise an error if any row has NaN in `repertoire_id`.
   - For each unique `repertoire_id` value in the file:
     - If it is in the metadata set → write rows to output as `part_table_<participant>_<specimen>.tsv.gz`.
     - If it is not in the metadata set → skip and record as skipped.
3. Build `specimen_overview.csv` for all 703 specimens (written and skipped), with metadata columns joined for those in metadata.
4. Write the text summary report.
5. Validate that all expected output files were created.

## Usage

```
python scripts/clean_airr_split_by_specimen.py
```

No arguments — all paths are hardcoded relative to the script's grandparent directory (`data_clean/../` = `Mal-ID/`).
