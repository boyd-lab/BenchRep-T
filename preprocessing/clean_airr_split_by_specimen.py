#!/usr/bin/env python3
"""
Split AIRR-format participant tables by specimen (repertoire_id),
keeping only specimens that appear in the metadata.

Reads every gzip-compressed AIRR TSV file from airr_format_clean/TCR/,
splits rows by the repertoire_id column, and writes one output file per
unique repertoire_id value — but only for specimens present in metadata.tsv.
Specimens found in the AIRR data but absent from the metadata are skipped
and reported.

Output filename convention:
    part_table_<participant>_<repertoire_id>.tsv.gz

All output files (specimen TSVs, specimen_overview.csv, summary report, log)
are written to scripts/reports/clean_airr_split_by_specimen/.

Usage:
    python scripts/clean_airr_split_by_specimen.py
"""

import gzip
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
MAL_ID_ROOT = SCRIPT_DIR.parent.parent  # scripts/ -> data_clean/ -> Mal-ID/

INPUT_DIR    = MAL_ID_ROOT / "data_clean" / "airr_format_clean" / "TCR"
METADATA_PATH = MAL_ID_ROOT / "data_clean" / "metadata.tsv"
REPORTS_DIR  = SCRIPT_DIR / "reports" / "clean_airr_split_by_specimen"
OUTPUT_DIR   = MAL_ID_ROOT / "data_clean" / "airr_format_clean_by_specimen" / "TCR"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = REPORTS_DIR / f"split_by_specimen_log_{timestamp}.txt"

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
    logger.info("AIRR SPLIT BY SPECIMEN")
    logger.info(f"Started: {timestamp}")
    logger.info("=" * 60)
    logger.info(f"Input:    {INPUT_DIR}")
    logger.info(f"Output:   {OUTPUT_DIR}")
    logger.info(f"Metadata: {METADATA_PATH}")
    logger.info(f"Reports:  {REPORTS_DIR}")

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return 1
    if not METADATA_PATH.exists():
        logger.error(f"Metadata file not found: {METADATA_PATH}")
        return 1

    # ------------------------------------------------------------------
    # Load metadata: build set of known specimen_labels
    # ------------------------------------------------------------------
    meta = pd.read_csv(METADATA_PATH, sep="\t")
    meta_specimen_labels = set(meta["specimen_label"].dropna().unique())
    logger.info(f"\nMetadata: {len(meta)} rows, {len(meta_specimen_labels)} unique specimen_labels\n")

    input_files = sorted(INPUT_DIR.glob("part_table_*.tsv.gz"))
    logger.info(f"Found {len(input_files)} input participant files\n")

    n_ok  = 0
    n_err = 0
    specimen_count_dist: Counter = Counter()  # n_written_specimens -> n_participant_files

    # Track every specimen encountered: {specimen_label: {participant, in_metadata, filename_written}}
    all_specimen_records: list[dict] = []
    expected_outputs: list[Path] = []

    for idx, input_path in enumerate(input_files, 1):
        participant = input_path.name.replace("part_table_", "").replace(".tsv.gz", "")

        if idx % 50 == 0 or idx == 1:
            logger.info(f"[{idx}/{len(input_files)}] {participant}")

        try:
            df = pd.read_csv(input_path, sep="\t", low_memory=False)

            n_na = df["repertoire_id"].isna().sum()
            if n_na > 0:
                raise ValueError(
                    f"{n_na} row(s) have NaN in 'repertoire_id' — cannot split."
                )

            unique_specimens = sorted(df["repertoire_id"].unique())
            n_written = 0

            for specimen in unique_specimens:
                in_meta = specimen in meta_specimen_labels
                safe_specimen = str(specimen).replace("/", "_").replace(" ", "_")
                out_name = f"part_table_{participant}_{safe_specimen}.tsv.gz"
                out_path = OUTPUT_DIR / out_name

                record = {
                    "participant_label": participant,
                    "specimen_label": specimen,
                    "in_metadata": in_meta,
                    "output_filename": out_name if in_meta else None,
                }
                all_specimen_records.append(record)

                if in_meta:
                    subset = df[df["repertoire_id"] == specimen].copy()
                    with gzip.open(out_path, "wt") as f:
                        subset.to_csv(f, sep="\t", index=False)
                    expected_outputs.append(out_path)
                    n_written += 1
                else:
                    logger.info(
                        f"  SKIP {specimen} (not in metadata)"
                    )

            specimen_count_dist[n_written] += 1

            n_skipped = len(unique_specimens) - n_written
            if n_written != len(unique_specimens):
                logger.info(
                    f"  {participant}: {n_written} written, {n_skipped} skipped "
                    f"(not in metadata)"
                )
            elif n_written > 1:
                logger.info(
                    f"  {participant}: split into {n_written} specimen tables "
                    f"({', '.join(unique_specimens)})"
                )
            else:
                logger.info(
                    f"  {participant}: 1 specimen ({unique_specimens[0]})"
                )

            n_ok += 1

        except Exception as e:
            logger.error(f"  ERROR processing {input_path.name}: {e}", exc_info=True)
            n_err += 1

    # ------------------------------------------------------------------
    # Build specimen_overview.csv (all specimens, with metadata joined)
    # ------------------------------------------------------------------
    overview_df = pd.DataFrame(all_specimen_records)
    # Join all metadata columns
    overview_df = overview_df.merge(
        meta, on=["participant_label", "specimen_label"], how="left"
    )
    # Fix in_metadata flag (merge may reorder)
    overview_df["in_metadata"] = overview_df["specimen_label"].isin(meta_specimen_labels)
    # Reorder columns: identifiers first
    leading = ["participant_label", "specimen_label", "in_metadata", "output_filename"]
    meta_cols = [c for c in meta.columns if c not in ["participant_label", "specimen_label"]]
    overview_df = overview_df[leading + meta_cols]

    overview_path = REPORTS_DIR / "specimen_overview.csv"
    overview_df.to_csv(overview_path, index=False)
    logger.info(f"\nSpecimen overview CSV: {overview_path}")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    n_total_specimens    = len(overview_df)
    n_in_meta            = overview_df["in_metadata"].sum()
    n_not_in_meta        = n_total_specimens - n_in_meta
    n_output_files       = len(expected_outputs)

    # Disease distribution for specimens written
    in_meta_df = overview_df[overview_df["in_metadata"]]
    disease_dist = in_meta_df["disease"].value_counts() if "disease" in in_meta_df.columns else {}

    # Study distribution for skipped specimens (not in metadata)
    skipped_df = overview_df[~overview_df["in_metadata"]]

    # ------------------------------------------------------------------
    # Write text summary report
    # ------------------------------------------------------------------
    report_path = REPORTS_DIR / f"split_by_specimen_report_{timestamp}.txt"
    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    h("AIRR SPLIT BY SPECIMEN — SUMMARY REPORT")
    lines.append(f"Generated : {timestamp}")
    lines.append(f"Input dir : {INPUT_DIR}")
    lines.append(f"Output dir: {OUTPUT_DIR}")
    lines.append(f"Metadata  : {METADATA_PATH}")

    h("OVERALL")
    lines.append(f"Participant files processed : {n_ok + n_err} ({n_err} errors)")
    lines.append(f"Total specimens in AIRR data: {n_total_specimens}")
    lines.append(f"  Written (in metadata)     : {n_in_meta}")
    lines.append(f"  Skipped (not in metadata) : {n_not_in_meta}")
    lines.append(f"Output specimen files created: {n_output_files}")

    h("WRITTEN SPECIMENS — DISEASE DISTRIBUTION")
    if hasattr(disease_dist, "items"):
        for disease, count in disease_dist.items():
            lines.append(f"  {disease}: {count} specimens")
    else:
        lines.append("  (disease column not found in metadata)")

    h("WRITTEN SPECIMENS — DISTRIBUTION PER PARTICIPANT FILE")
    lines.append("  (how many specimen files were written per input participant file)")
    for n_spec in sorted(specimen_count_dist):
        n_files = specimen_count_dist[n_spec]
        lines.append(f"  {n_spec} specimen(s) written: {n_files} participant file(s)")

    h("SKIPPED SPECIMENS (not in metadata)")
    lines.append(f"  Total skipped: {n_not_in_meta}")
    if n_not_in_meta > 0:
        lines.append("  Specimens:")
        for _, row in skipped_df.iterrows():
            lines.append(f"    {row['participant_label']}  /  {row['specimen_label']}")

    h("VALIDATION")
    missing_files = [p for p in expected_outputs if not p.exists()]
    if missing_files:
        lines.append(f"  MISSING {len(missing_files)}/{n_output_files} output files:")
        for p in missing_files:
            lines.append(f"    {p.name}")
    else:
        lines.append(f"  OK: all {n_output_files} output files present.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)
    lines.append("")

    report_path.write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Participant files processed : {n_ok + n_err} ({n_err} errors)")
    logger.info(f"  Total specimens in AIRR data: {n_total_specimens}")
    logger.info(f"    Written (in metadata)     : {n_in_meta}")
    logger.info(f"    Skipped (not in metadata) : {n_not_in_meta}")
    logger.info(f"  Output specimen files created: {n_output_files}")
    logger.info("")
    logger.info("  Specimens written per participant file:")
    for n_spec in sorted(specimen_count_dist):
        logger.info(f"    {n_spec} written: {specimen_count_dist[n_spec]} participant file(s)")

    logger.info("\nPost-run validation: verifying output files...")
    if missing_files:
        logger.error(f"  MISSING {len(missing_files)}/{n_output_files} output files:")
        for p in missing_files:
            logger.error(f"    {p.name}")
    else:
        logger.info(f"  OK: all {n_output_files} output files present.")

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE: {n_ok} participant files processed, {n_err} errors")
    logger.info(f"Output:           {OUTPUT_DIR}")
    logger.info(f"Specimen overview: {overview_path}")
    logger.info(f"Summary report:    {report_path}")
    logger.info(f"Log:               {log_path}")
    logger.info("=" * 60)

    return 0 if (n_err == 0 and not missing_files) else 1


if __name__ == "__main__":
    sys.exit(main())
