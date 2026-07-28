"""Prepare Emerson CMV data for Mal-ID-Lite.

This implements PIPELINE_GUIDE.md sections 4.1 and 4.2:

* one metadata row per labeled specimen, with participant_label,
  specimen_label, disease, CV_fold, and available_gene_loci;
* one gzip-compressed AIRR TSV per participant, named
  part_table_{participant_label}.tsv.gz;
* allele-bearing v_call/j_call values;
* repertoire_id values matching metadata specimen_label values;
* a nucleotide cdr3 column so Mal-ID-Lite can compute clone_id itself.

The existing Emerson preprocessing filters and gene harmonization are reused.
Rows whose trimmed nucleotide CDR3 is not valid ACGT, does not have the
expected length, or does not translate to cdr3_aa are excluded because they
cannot support nucleotide-based clone assignment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from preprocess_repertoires import preprocess_dataframe


SCRIPT_DIR = Path(__file__).resolve().parent
AIRR_BENCH_DIR = SCRIPT_DIR.parent
DEFAULT_CMV_DIR = AIRR_BENCH_DIR / "data" / "external_datasets" / "CMV"

RAW_COLUMNS = {
    "amino_acid",
    "v_gene",
    "v_allele",
    "j_gene",
    "j_allele",
    "v_resolved",
    "j_resolved",
    "cdr3_rearrangement",
    "rearrangement",
    "frame_type",
    "templates",
    "seq_reads",
}

OUTPUT_COLUMNS = [
    "repertoire_id",
    "v_call",
    "j_call",
    "cdr3_aa",
    "cdr3",
    "productive",
    "sequence",
    "num_reads",
    "participant_label",
]

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate_nt(sequence: str) -> str:
    return "".join(
        CODON_TABLE.get(sequence[i : i + 3], "X")
        for i in range(0, len(sequence), 3)
    )


def _extract_allele(
    resolved: object, separate_allele: object
) -> tuple[str, bool]:
    """Extract an AIRR allele, falling back to canonical *01 when unresolved.

    Some immuneACCESS rows resolve a gene but not an allele (for example,
    ``TCRBV19-01`` with a blank ``v_allele``). Mal-ID-Lite requires
    allele-bearing calls, so those rows use the canonical ``*01`` allele and
    are counted explicitly as imputed in the conversion report.
    """
    if isinstance(resolved, str):
        match = re.search(r"\*(\d+)$", resolved.strip())
        if match:
            return match.group(1).zfill(2), False

    if not pd.isna(separate_allele):
        text = str(separate_allele).strip()
        if re.fullmatch(r"\d+(?:\.0+)?", text):
            return str(int(float(text))).zfill(2), False
    return "01", True


def _append_alleles(
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    v_allele_results = [
        _extract_allele(resolved, separate)
        for resolved, separate in zip(
            processed["v_resolved"], processed["v_allele"]
        )
    ]
    j_allele_results = [
        _extract_allele(resolved, separate)
        for resolved, separate in zip(
            processed["j_resolved"], processed["j_allele"]
        )
    ]
    v_alleles = [allele for allele, _ in v_allele_results]
    j_alleles = [allele for allele, _ in j_allele_results]

    processed["v_call"] = [
        f"{gene}*{allele}"
        for gene, allele in zip(processed["v_call"], v_alleles)
    ]
    processed["j_call"] = [
        f"{gene}*{allele}"
        for gene, allele in zip(processed["j_call"], j_alleles)
    ]
    return processed, {
        "n_v_alleles_imputed_as_01": sum(
            imputed for _, imputed in v_allele_results
        ),
        "n_j_alleles_imputed_as_01": sum(
            imputed for _, imputed in j_allele_results
        ),
    }


def _add_and_validate_cdr3_nt(
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Trim conserved first/last codons and retain clone-computable rows."""
    raw_cdr3 = processed["cdr3_rearrangement"].astype("string").str.upper()
    cdr3_nt = raw_cdr3.str.slice(3, -3)

    valid_acgt = cdr3_nt.str.fullmatch(r"[ACGT]+", na=False)
    valid_length = cdr3_nt.str.len().eq(processed["cdr3_aa"].str.len() * 3)
    translated = cdr3_nt.fillna("").map(_translate_nt)
    valid_translation = translated.eq(processed["cdr3_aa"])

    keep = valid_acgt & valid_length & valid_translation
    stats = {
        "n_invalid_cdr3_nt_acgt": int((~valid_acgt).sum()),
        "n_invalid_cdr3_nt_length": int((valid_acgt & ~valid_length).sum()),
        "n_cdr3_nt_translation_mismatch": int(
            (valid_acgt & valid_length & ~valid_translation).sum()
        ),
        "n_cdr3_nt_rows_dropped": int((~keep).sum()),
    }

    processed = processed.loc[keep].copy()
    processed["cdr3"] = cdr3_nt.loc[keep]
    return processed, stats


def _fill_missing_num_reads(
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Use Emerson read counts when template counts are unavailable."""
    missing_templates = processed["num_reads"].isna()
    processed.loc[missing_templates, "num_reads"] = processed.loc[
        missing_templates, "seq_reads"
    ]

    n_still_missing = int(processed["num_reads"].isna().sum())
    if n_still_missing:
        raise ValueError(
            f"{n_still_missing} sequence row(s) have neither templates nor seq_reads"
        )

    return processed, int(missing_templates.sum())


def convert_participant(
    raw_path: str,
    output_path: str,
    participant_label: str,
    specimen_label: str,
    chunksize: int,
) -> dict[str, object]:
    """Convert one Emerson participant file atomically."""
    raw_path_obj = Path(raw_path)
    output_path_obj = Path(output_path)
    temp_path = output_path_obj.with_name(
        f".{output_path_obj.name}.tmp.{os.getpid()}"
    )

    totals: dict[str, object] = {
        "participant_label": participant_label,
        "specimen_label": specimen_label,
        "input_file": raw_path_obj.name,
        "output_file": output_path_obj.name,
        "n_in": 0,
        "n_after_existing_preprocessing": 0,
        "n_out": 0,
        "n_non_productive": 0,
        "n_unresolved": 0,
        "n_missing_field": 0,
        "n_invalid_cdr3": 0,
        "n_family_only": 0,
        "n_invalid_cdr3_nt_acgt": 0,
        "n_invalid_cdr3_nt_length": 0,
        "n_cdr3_nt_translation_mismatch": 0,
        "n_cdr3_nt_rows_dropped": 0,
        "n_v_alleles_imputed_as_01": 0,
        "n_j_alleles_imputed_as_01": 0,
        "n_num_reads_from_seq_reads": 0,
    }

    first_chunk = True
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as handle:
            chunks = pd.read_csv(
                raw_path_obj,
                sep="\t",
                usecols=lambda column: column in RAW_COLUMNS,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in chunks:
                missing_raw = sorted(RAW_COLUMNS - set(chunk.columns))
                if missing_raw:
                    raise ValueError(
                        f"{raw_path_obj.name}: missing raw columns {missing_raw}"
                    )

                processed, existing_stats = preprocess_dataframe(
                    chunk,
                    input_basename=raw_path_obj.name,
                    repertoire_id=specimen_label,
                    participant_label=participant_label,
                    compact_output=False,
                )
                processed, n_num_reads_from_seq_reads = _fill_missing_num_reads(
                    processed
                )
                processed, allele_stats = _append_alleles(processed)
                processed, nt_stats = _add_and_validate_cdr3_nt(processed)
                processed["productive"] = "T"
                processed = processed[OUTPUT_COLUMNS]

                processed.to_csv(
                    handle,
                    sep="\t",
                    index=False,
                    header=first_chunk,
                )
                first_chunk = False

                totals["n_in"] += existing_stats["n_in"]
                totals["n_after_existing_preprocessing"] += existing_stats["n_out"]
                totals["n_out"] += len(processed)
                totals["n_num_reads_from_seq_reads"] += n_num_reads_from_seq_reads
                for key in (
                    "n_non_productive",
                    "n_unresolved",
                    "n_missing_field",
                    "n_invalid_cdr3",
                    "n_family_only",
                ):
                    totals[key] += existing_stats[key]
                for key, value in nt_stats.items():
                    totals[key] += value
                for key, value in allele_stats.items():
                    totals[key] += value

        if first_chunk:
            raise ValueError(f"{raw_path_obj.name}: no readable chunks")
        if totals["n_out"] == 0:
            raise ValueError(f"{raw_path_obj.name}: no sequences survived conversion")
        os.replace(temp_path, output_path_obj)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return totals


def create_malid_metadata(source_metadata: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(source_metadata, sep="\t")
    required = {
        "participant_label",
        "specimen_label",
        "disease",
        "malid_cross_validation_fold_id_when_in_test_set",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Source metadata is missing required columns: {missing}")

    excluded = metadata.loc[metadata["disease"].isna()].copy()
    metadata = metadata.loc[metadata["disease"].notna()].copy()
    metadata = metadata.rename(
        columns={
            "malid_cross_validation_fold_id_when_in_test_set": "CV_fold"
        }
    )
    metadata["available_gene_loci"] = "TCRB"
    metadata["repertoire_file"] = metadata["participant_label"].map(
        lambda participant: f"part_table_{participant}.tsv.gz"
    )

    first_columns = [
        "participant_label",
        "specimen_label",
        "disease",
        "CV_fold",
        "available_gene_loci",
    ]
    remaining = [column for column in metadata.columns if column not in first_columns]
    metadata = metadata[first_columns + remaining]

    for column in first_columns:
        if metadata[column].isna().any():
            raise ValueError(f"Output metadata column {column!r} contains NaN")
    if metadata["specimen_label"].duplicated().any():
        raise ValueError("Output metadata specimen_label values are not unique")
    if (metadata.groupby("participant_label")["disease"].nunique() != 1).any():
        raise ValueError("A participant has more than one disease label")
    if (metadata.groupby("participant_label")["CV_fold"].nunique() != 1).any():
        raise ValueError("A participant has more than one CV_fold")

    metadata["CV_fold"] = metadata["CV_fold"].astype(int)
    return metadata, excluded


def validate_outputs(output_dir: Path, metadata: pd.DataFrame) -> dict[str, object]:
    expected_participants = set(metadata["participant_label"].astype(str))
    output_files = sorted(output_dir.glob("part_table_*.tsv.gz"))
    observed_participants = {
        path.name.removeprefix("part_table_").removesuffix(".tsv.gz")
        for path in output_files
    }
    if observed_participants != expected_participants:
        raise ValueError(
            "Participant file set does not match metadata: "
            f"missing={sorted(expected_participants - observed_participants)[:10]}, "
            f"extra={sorted(observed_participants - expected_participants)[:10]}"
        )

    required_sequence_columns = {
        "repertoire_id",
        "v_call",
        "j_call",
        "cdr3_aa",
        "cdr3",
        "num_reads",
    }
    total_rows = 0
    for path in output_files:
        header = pd.read_csv(path, sep="\t", nrows=0)
        missing = required_sequence_columns - set(header.columns)
        if missing:
            raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

        participant = path.name.removeprefix("part_table_").removesuffix(".tsv.gz")
        expected_specimens = set(
            metadata.loc[
                metadata["participant_label"].astype(str).eq(participant),
                "specimen_label",
            ].astype(str)
        )
        participant_rows = 0
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=OUTPUT_COLUMNS,
            chunksize=250_000,
            low_memory=False,
        ):
            participant_rows += len(chunk)
            if chunk[list(required_sequence_columns)].isna().any().any():
                raise ValueError(f"{path.name}: NaN in required sequence columns")
            if set(chunk["repertoire_id"].astype(str)) != expected_specimens:
                raise ValueError(
                    f"{path.name}: repertoire_id does not match metadata specimen_label"
                )
            if not chunk["v_call"].str.fullmatch(r"[^*]+\*\d+", na=False).all():
                raise ValueError(f"{path.name}: v_call without allele")
            if not chunk["j_call"].str.fullmatch(r"[^*]+\*\d+", na=False).all():
                raise ValueError(f"{path.name}: j_call without allele")
            if not chunk["cdr3"].str.fullmatch(r"[ACGT]+", na=False).all():
                raise ValueError(f"{path.name}: invalid nucleotide cdr3")
            if not (
                chunk["cdr3"].str.len() == chunk["cdr3_aa"].str.len() * 3
            ).all():
                raise ValueError(f"{path.name}: cdr3/cdr3_aa length mismatch")
            translated = chunk["cdr3"].map(_translate_nt)
            if not translated.eq(chunk["cdr3_aa"]).all():
                raise ValueError(f"{path.name}: cdr3 translation mismatch")
            if not chunk["productive"].eq("T").all():
                raise ValueError(f"{path.name}: nonproductive output row")
        if participant_rows == 0:
            raise ValueError(f"{path.name}: output file is empty")
        total_rows += participant_rows

    return {
        "n_metadata_rows": len(metadata),
        "n_participants": metadata["participant_label"].nunique(),
        "n_specimens": metadata["specimen_label"].nunique(),
        "n_sequence_files": len(output_files),
        "n_sequence_rows": total_rows,
        "fold_counts": {
            str(key): int(value)
            for key, value in metadata["CV_fold"].value_counts().sort_index().items()
        },
        "disease_counts": {
            str(key): int(value)
            for key, value in metadata["disease"].value_counts().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_CMV_DIR / "raw",
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=DEFAULT_CMV_DIR / "emerson_cohort_metadata.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CMV_DIR / "processed_for_malid",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep nonempty participant outputs that already exist.",
    )
    args = parser.parse_args()

    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be >= 1")

    metadata, excluded = create_malid_metadata(args.source_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.tsv"
    excluded_path = args.output_dir / "excluded_unknown_cmv_status.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    excluded.to_csv(excluded_path, sep="\t", index=False)

    jobs: list[tuple[str, str, str, str, int]] = []
    for row in metadata.itertuples(index=False):
        participant = str(row.participant_label)
        specimen = str(row.specimen_label)
        raw_path = args.raw_dir / f"{specimen}.tsv"
        output_path = args.output_dir / f"part_table_{participant}.tsv.gz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw repertoire: {raw_path}")
        if (
            args.skip_existing
            and output_path.is_file()
            and output_path.stat().st_size > 0
        ):
            continue
        jobs.append(
            (
                str(raw_path),
                str(output_path),
                participant,
                specimen,
                args.chunksize,
            )
        )

    print(f"Metadata: {len(metadata)} labeled specimens")
    print(f"Excluded: {len(excluded)} specimens with unknown CMV status")
    print(f"Converting: {len(jobs)} participant files with {args.n_jobs} workers")

    stats: list[dict[str, object]] = []
    if args.n_jobs == 1:
        for index, job in enumerate(jobs, start=1):
            stats.append(convert_participant(*job))
            if index % 10 == 0 or index == len(jobs):
                print(f"  Processed {index}/{len(jobs)} files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            future_to_job = {
                executor.submit(convert_participant, *job): job for job in jobs
            }
            for index, future in enumerate(as_completed(future_to_job), start=1):
                stats.append(future.result())
                if index % 10 == 0 or index == len(jobs):
                    print(f"  Processed {index}/{len(jobs)} files", flush=True)

    stats_path = args.output_dir / "conversion_stats.tsv"
    if stats:
        stats_df = pd.DataFrame(stats).sort_values("participant_label")
        stats_df.to_csv(stats_path, sep="\t", index=False)
    elif not stats_path.exists():
        raise ValueError(
            "No files converted and conversion_stats.tsv does not exist; "
            "cannot summarize a resumed conversion"
        )

    validation = validate_outputs(args.output_dir, metadata)
    stats_df = pd.read_csv(stats_path, sep="\t")
    conversion_totals = {
        column: int(stats_df[column].sum())
        for column in stats_df.columns
        if column.startswith("n_")
    }
    report = {
        **validation,
        "excluded_unknown_cmv_status": len(excluded),
        "conversion_totals": conversion_totals,
        "known_deviations_from_pipeline_guide": [
            {
                "column": "v_call/j_call allele",
                "reason": (
                    f"The immuneACCESS source lacks an allele assignment for "
                    f"{conversion_totals['n_v_alleles_imputed_as_01']} V calls "
                    f"and {conversion_totals['n_j_alleles_imputed_as_01']} J calls. "
                    "To satisfy Mal-ID-Lite's allele-bearing call format without "
                    "discarding resolved gene calls, these use canonical *01. "
                    "This allele is imputed, not observed."
                ),
            },
            {
                "column": "v_score",
                "reason": (
                    "The Emerson immuneACCESS source has no V alignment score. "
                    "The column is omitted, so Mal-ID-Lite will emit its documented "
                    "warning and skip the v_score > 80 filter."
                ),
            },
            {
                "column": "replicate_label",
                "reason": (
                    "No replicate identifier exists in the Emerson source. "
                    "The column is omitted, so Mal-ID-Lite will emit its documented "
                    "warning and skip sequence+replicate deduplication."
                ),
            },
        ],
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Metadata written: {metadata_path}")
    print(f"Excluded rows: {excluded_path}")
    print(f"Conversion stats: {stats_path}")
    print(f"Validation report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
