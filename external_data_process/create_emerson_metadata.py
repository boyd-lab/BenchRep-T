"""Create package-compatible metadata for the Emerson et al. CMV cohort.

The official Supplementary Table 1 has 666 Cohort 1 rows and 120 Cohort 2
rows.  In the immuneACCESS download used by AIRR Bench, sorted ``P*.tsv``
files correspond row-for-row to Cohort 1, while Cohort 2 subject IDs map
directly to ``Keck*_MC1.tsv`` filenames.

Fold policy:
  * Original Emerson Cohort 2 is always fold 2.
  * Cohort 1 is divided reproducibly between folds 0 and 1, stratified on
    known CMV status.
  * Subjects whose official CMV status is ``Unknown`` remain in the metadata
    with a blank disease value. Disease evaluators exclude these rows.

Example:
    python external_data_process/create_emerson_metadata.py \
        --supplement-xlsx data/external_datasets/CMV/emerson_supplementary_table_1.xlsx \
        --raw-dir data/external_datasets/CMV/raw \
        --processed-dir data/external_datasets/CMV/processed \
        --output data/external_datasets/CMV/emerson_cohort_metadata.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FOLD_COLUMN = "malid_cross_validation_fold_id_when_in_test_set"
HEALTHY_LABEL = "Healthy/Background"
TARGET_DISEASE = "CMV"
RANDOM_SEED = 7


def _clean_source_value(value: object) -> object:
    """Convert source ``Unknown`` values to missing values."""
    if pd.isna(value) or str(value).strip() == "Unknown":
        return pd.NA
    return value


def _split_race_ethnicity(value: object) -> tuple[object, object]:
    """Split the supplement's combined race/ethnicity field."""
    value = _clean_source_value(value)
    if pd.isna(value):
        return pd.NA, pd.NA

    text = str(value).strip()
    if ", " not in text:
        return text, pd.NA
    race, ethnicity = text.split(", ", maxsplit=1)
    return race, ethnicity


def _to_ancestry(race: object, ethnicity: object) -> object:
    """Map source categories to the ancestry vocabulary used by AIRR Bench."""
    if not pd.isna(ethnicity) and str(ethnicity) == "Hispanic or Latino":
        return "Hispanic/Latino"
    if pd.isna(race):
        return pd.NA

    return {
        "White": "Caucasian",
        "Black or African American": "African",
        "Asian": "Asian",
        "Native Hawaiian or other Pacific Islander": "Asian",
    }.get(str(race), pd.NA)


def _read_raw_tags(path: Path) -> dict[str, object]:
    """Read sample-level fields from the first row of an immuneACCESS file."""
    row = pd.read_csv(
        path,
        sep="\t",
        nrows=1,
        usecols=["sample_name", "sample_tags"],
        low_memory=False,
    ).iloc[0]
    tags = str(row["sample_tags"])

    def extract(pattern: str) -> object:
        match = re.search(pattern, tags)
        return match.group(1) if match else pd.NA

    return {
        "sample_name": str(row["sample_name"]),
        "sex": extract(r"Biological Sex:([^,]+)"),
        "cmv_status": extract(r"Virus Diseases:Cytomegalovirus ([+-])"),
        "age": extract(r"Age:(\d+) Years"),
    }


def _validate_source_mapping(
    source: pd.DataFrame, raw_files: list[Path], cohort: int
) -> None:
    """Guard against silently pairing a supplement row with the wrong file."""
    if len(source) != len(raw_files):
        raise ValueError(
            f"Cohort {cohort}: {len(source)} supplement rows but "
            f"{len(raw_files)} raw files"
        )

    mismatches: list[str] = []
    for (_, row), raw_path in zip(source.iterrows(), raw_files):
        raw = _read_raw_tags(raw_path)
        if raw["sample_name"] != raw_path.stem:
            mismatches.append(
                f"{raw_path.name}: internal sample name is {raw['sample_name']!r}"
            )

        if cohort == 2:
            expected_stem = f"{row['Subject ID']}_MC1"
            if raw_path.stem != expected_stem:
                mismatches.append(
                    f"{raw_path.name}: expected filename stem {expected_stem!r}"
                )

        source_sex = _clean_source_value(row["Sex"])
        if not pd.isna(raw["sex"]) and not pd.isna(source_sex):
            if str(raw["sex"]) != str(source_sex):
                mismatches.append(
                    f"{raw_path.name}: sex {raw['sex']!r} != {source_sex!r}"
                )

        source_status = _clean_source_value(row["Known CMV status"])
        if not pd.isna(raw["cmv_status"]) and not pd.isna(source_status):
            if str(raw["cmv_status"]) != str(source_status):
                mismatches.append(
                    f"{raw_path.name}: CMV status {raw['cmv_status']!r} "
                    f"!= {source_status!r}"
                )

        # Three Cohort 1 raw age tags differ from the supplement by 2 years.
        # A tolerance catches ordering errors while allowing those source-data
        # discrepancies and normal rounding from fractional ages.
        source_age = pd.to_numeric(
            pd.Series([_clean_source_value(row["Age"])]), errors="coerce"
        ).iloc[0]
        if not pd.isna(raw["age"]) and not pd.isna(source_age):
            if abs(int(raw["age"]) - round(float(source_age))) > 2:
                mismatches.append(
                    f"{raw_path.name}: age {raw['age']!r} != {source_age!r}"
                )

    if mismatches:
        details = "\n  ".join(mismatches[:20])
        raise ValueError(
            f"Source-to-file mapping validation failed ({len(mismatches)} issues):"
            f"\n  {details}"
        )


def _records_for_cohort(
    source: pd.DataFrame, raw_files: list[Path], cohort: int
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    cohort_name = "discovery" if cohort == 1 else "validation"

    for (_, row), raw_path in zip(source.iterrows(), raw_files):
        status = _clean_source_value(row["Known CMV status"])
        disease = {
            "+": TARGET_DISEASE,
            "-": HEALTHY_LABEL,
        }.get(status, pd.NA)
        subtype = {
            "+": "CMV seropositive",
            "-": "CMV seronegative",
        }.get(status, pd.NA)

        race, ethnicity = _split_race_ethnicity(row["Race and ethnicity "])
        age = pd.to_numeric(
            pd.Series([_clean_source_value(row["Age"])]), errors="coerce"
        ).iloc[0]
        sex = {
            "Female": "F",
            "Male": "M",
        }.get(_clean_source_value(row["Sex"]), pd.NA)

        records.append(
            {
                "participant_label": raw_path.stem,
                "specimen_label": raw_path.stem,
                "disease": disease,
                "specimen_time_point": pd.NA,
                "study_name": "Emerson et al. 2017",
                "available_gene_loci": "GeneLocus.TCR",
                "disease_subtype": subtype,
                "age": age,
                "sex": sex,
                "ancestry": _to_ancestry(race, ethnicity),
                FOLD_COLUMN: 2 if cohort == 2 else pd.NA,
                "repertoire_id": raw_path.stem,
                "repertoire_file": raw_path.name,
                "emerson_subject_id": str(row["Subject ID"]),
                "cohort": cohort,
                "cohort_name": cohort_name,
                "race": race,
                "ethnicity": ethnicity,
                "race_and_ethnicity": _clean_source_value(
                    row["Race and ethnicity "]
                ),
                "known_cmv_status": status,
                "metadata_source": "Emerson Supplementary Table 1",
            }
        )

    return pd.DataFrame.from_records(records)


def _assign_discovery_folds(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratify Cohort 1 across folds 0/1 and keep their total sizes equal."""
    result = metadata.copy()
    discovery = result["cohort"].eq(1)
    known = discovery & result["disease"].notna()
    unknown = discovery & result["disease"].isna()
    rng = np.random.default_rng(seed)

    # Alternate shuffled members of each class. With 289 CMV-positive and 352
    # CMV-negative subjects, this gives known fold sizes 321 and 320.
    for disease in [TARGET_DISEASE, HEALTHY_LABEL]:
        indices = result.index[
            known & result["disease"].eq(disease)
        ].to_numpy(copy=True)
        rng.shuffle(indices)
        result.loc[indices[::2], FOLD_COLUMN] = 0
        result.loc[indices[1::2], FOLD_COLUMN] = 1

    # Unknown labels do not participate in stratification. Allocate each one
    # to the currently smaller fold so the full Cohort 1 split is 333/333.
    unknown_indices = result.index[unknown].to_numpy(copy=True)
    rng.shuffle(unknown_indices)
    fold_counts = result.loc[known, FOLD_COLUMN].value_counts().to_dict()
    for index in unknown_indices:
        fold = 0 if fold_counts.get(0, 0) < fold_counts.get(1, 0) else 1
        result.loc[index, FOLD_COLUMN] = fold
        fold_counts[fold] = fold_counts.get(fold, 0) + 1

    result[FOLD_COLUMN] = result[FOLD_COLUMN].astype(int)
    return result


def create_metadata(
    supplement_xlsx: Path,
    raw_dir: Path,
    processed_dir: Path | None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    cohort1 = pd.read_excel(supplement_xlsx, sheet_name="Cohort 1")
    cohort2 = pd.read_excel(supplement_xlsx, sheet_name="Cohort 2")
    cohort1_files = sorted(raw_dir.glob("P*.tsv"))
    cohort2_files = sorted(raw_dir.glob("Keck*.tsv"))

    _validate_source_mapping(cohort1, cohort1_files, cohort=1)
    _validate_source_mapping(cohort2, cohort2_files, cohort=2)

    metadata = pd.concat(
        [
            _records_for_cohort(cohort1, cohort1_files, cohort=1),
            _records_for_cohort(cohort2, cohort2_files, cohort=2),
        ],
        ignore_index=True,
    )
    metadata = _assign_discovery_folds(metadata, seed=seed)

    if processed_dir is not None:
        missing = [
            name
            for name in metadata["repertoire_file"]
            if not (processed_dir / name).is_file()
            or (processed_dir / name).stat().st_size == 0
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} processed repertoire files are missing or empty; "
                f"first examples: {missing[:10]}"
            )

    if metadata["participant_label"].duplicated().any():
        raise ValueError("participant_label values must be unique")
    if set(metadata[FOLD_COLUMN].unique()) != {0, 1, 2}:
        raise ValueError("Expected fold IDs 0, 1, and 2")
    if not metadata.loc[metadata["cohort"].eq(2), FOLD_COLUMN].eq(2).all():
        raise ValueError("Every Cohort 2 participant must be assigned to fold 2")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-xlsx", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    metadata = create_metadata(
        supplement_xlsx=args.supplement_xlsx,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, sep="\t", index=False)

    print(f"Wrote {len(metadata)} rows to {args.output}")
    print("\nCohort counts:")
    print(metadata["cohort"].value_counts().sort_index().to_string())
    print("\nFold counts:")
    print(metadata[FOLD_COLUMN].value_counts().sort_index().to_string())
    print("\nDisease by fold (blank disease = official status Unknown):")
    display = metadata.assign(disease=metadata["disease"].fillna("<blank>"))
    print(pd.crosstab(display[FOLD_COLUMN], display["disease"]).to_string())
    print("\nDemographic completeness:")
    print(metadata[["age", "sex", "ancestry"]].notna().sum().to_string())


if __name__ == "__main__":
    main()
