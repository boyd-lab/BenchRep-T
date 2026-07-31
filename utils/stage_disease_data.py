"""Create a uniform, symlink-only view of native Hugging Face cohorts."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from utils.huggingface_data import COHORTS


CANONICAL_FOLD_COLUMN = "malid_cross_validation_fold_id_when_in_test_set"
CANONICAL_CONTROL = "Healthy/Background"


@dataclass(frozen=True)
class DiseaseCohort:
    targets: tuple[str, ...]
    control: str
    fold_column: str


DISEASE_COHORTS: dict[str, DiseaseCohort] = {
    "malid": DiseaseCohort(
        ("HIV", "T1D", "Lupus", "Covid19", "Influenza"),
        "Healthy/Background",
        CANONICAL_FOLD_COLUMN,
    ),
    "mitchell-t1d": DiseaseCohort(
        ("T1D",), "Healthy/Background", CANONICAL_FOLD_COLUMN
    ),
    "rawat-t1d": DiseaseCohort(("T1D",), "Healthy/Background", "CV_fold"),
    "tb": DiseaseCohort(("Progressor",), "Controller", "CV_fold"),
    "ra": DiseaseCohort(("Rheumatoid Arthritis",), "Healthy", "CV_fold"),
    "cmv": DiseaseCohort(("CMV",), "Healthy/Background", CANONICAL_FOLD_COLUMN),
}


def stage_cohort(data_root: Path, output_root: Path, name: str) -> tuple[Path, Path]:
    """Normalize metadata and link native files under canonical evaluator names."""
    native = COHORTS[name]
    disease = DISEASE_COHORTS[name]
    source_metadata = data_root / native.metadata
    source_repertoires = data_root / native.repertoires
    if not source_metadata.is_file():
        raise FileNotFoundError(f"Missing metadata: {source_metadata}")
    if not source_repertoires.is_dir():
        raise FileNotFoundError(f"Missing repertoire directory: {source_repertoires}")

    cohort_output = output_root / name
    repertoire_output = cohort_output / "repertoires"
    repertoire_output.mkdir(parents=True, exist_ok=True)
    metadata_output = cohort_output / "metadata.tsv"

    with source_metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if disease.fold_column not in fieldnames:
        raise ValueError(
            f"{source_metadata} lacks fold column {disease.fold_column!r}"
        )
    if CANONICAL_FOLD_COLUMN not in fieldnames:
        fieldnames.append(CANONICAL_FOLD_COLUMN)

    for row in rows:
        row[CANONICAL_FOLD_COLUMN] = row[disease.fold_column]
        if row["disease"] == disease.control:
            row["disease"] = CANONICAL_CONTROL

        source_name = native.file_template.format(**row)
        source = (source_repertoires / source_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing repertoire: {source}")
        destination = repertoire_output / (
            f"part_table_{row['participant_label']}_{row['specimen_label']}.tsv.gz"
        )
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing staged path: {destination}"
            )
        relative_source = Path(os.path.relpath(source, start=destination.parent))
        destination.symlink_to(relative_source)

    with metadata_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return metadata_output, repertoire_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        action="append",
        choices=tuple(DISEASE_COHORTS) + ("all",),
        help="Cohort to stage; repeat as needed (default: all).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = args.cohort or ["all"]
    names = tuple(DISEASE_COHORTS) if "all" in selected else tuple(selected)
    for name in dict.fromkeys(names):
        metadata, repertoires = stage_cohort(
            args.data_root.resolve(), args.output_root.resolve(), name
        )
        targets = ", ".join(DISEASE_COHORTS[name].targets)
        print(
            f"{name}: metadata={metadata}, repertoires={repertoires}, "
            f"targets={targets}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
