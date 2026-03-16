#!/usr/bin/env python3
"""
Split AIRR-format participant tables by specimen (repertoire_id).

Reads every gzip-compressed AIRR TSV file from airr_format_clean/TCR/,
splits rows by the repertoire_id column, and writes one output file per
unique repertoire_id value to airr_format_clean_by_specimen/TCR/.

Output filename convention:
    part_table_<participant>_<repertoire_id>.tsv.gz

Summary and validation are printed to stdout and written to a log file
under scripts/reports/.

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

SCRIPT_DIR = Path(__file__).parent
MAL_ID_ROOT = SCRIPT_DIR.parent.parent  # scripts/ -> data_clean/ -> Mal-ID/

INPUT_DIR  = MAL_ID_ROOT / "data_clean" / "airr_format_clean" / "TCR"
OUTPUT_DIR = MAL_ID_ROOT / "data_clean" / "airr_format_clean_by_specimen" / "TCR"
REPORTS_DIR = SCRIPT_DIR / "reports"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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
    logger.info(f"Input:  {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_DIR.glob("part_table_*.tsv.gz"))
    logger.info(f"\nFound {len(input_files)} input files\n")

    n_ok = 0
    n_err = 0
    specimen_count_dist: Counter = Counter()   # n_specimens -> n_participant_files
    expected_outputs: list[Path] = []          # all output paths that should be created

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
            unique_specimens = df["repertoire_id"].unique()
            n_specimens = len(unique_specimens)
            specimen_count_dist[n_specimens] += 1

            for specimen in sorted(unique_specimens):
                # Sanitize specimen value for use in a filename
                safe_specimen = str(specimen).replace("/", "_").replace(" ", "_")
                out_name = f"part_table_{participant}_{safe_specimen}.tsv.gz"
                out_path = OUTPUT_DIR / out_name
                expected_outputs.append(out_path)

                subset = df[df["repertoire_id"] == specimen].copy()
                with gzip.open(out_path, "wt") as f:
                    subset.to_csv(f, sep="\t", index=False)

            if n_specimens > 1:
                logger.info(
                    f"  {participant}: split into {n_specimens} specimen tables "
                    f"({', '.join(sorted(str(s) for s in unique_specimens))})"
                )
            else:
                logger.info(
                    f"  {participant}: 1 specimen "
                    f"({unique_specimens[0] if n_specimens else 'none'})"
                )

            n_ok += 1

        except Exception as e:
            logger.error(f"  ERROR processing {input_path.name}: {e}", exc_info=True)
            n_err += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_output_files = len(expected_outputs)
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Participant files processed : {n_ok + n_err}")
    logger.info(f"  Errors                      : {n_err}")
    logger.info(f"  Total specimen files created: {total_output_files}")
    logger.info("")
    logger.info("  Distribution (specimens per participant file):")
    for n_spec in sorted(specimen_count_dist):
        n_files = specimen_count_dist[n_spec]
        logger.info(f"    {n_spec} specimen(s): {n_files} participant file(s)")

    # ------------------------------------------------------------------
    # Post-run validation: verify all expected output files were created
    # ------------------------------------------------------------------
    logger.info("\nPost-run validation: verifying output files...")
    missing = [p for p in expected_outputs if not p.exists()]
    if missing:
        logger.error(f"  MISSING {len(missing)}/{total_output_files} output files:")
        for p in missing:
            logger.error(f"    {p.name}")
    else:
        logger.info(f"  OK: all {total_output_files} output files present.")

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE: {n_ok} participant files processed, {n_err} errors")
    logger.info(f"Output:  {OUTPUT_DIR}")
    logger.info(f"Log:     {log_path}")
    logger.info("=" * 60)

    return 0 if (n_err == 0 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
