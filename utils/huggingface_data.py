"""Download the private BenchRep-T dataset into the benchmark's data layout.

The model evaluators intentionally continue to consume ordinary files.  This
module materializes only the cohorts and task assets requested by the user so
that every existing evaluator can run without knowing about Hugging Face.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_REPO_ID = "neurips-2026-dataset/BenchRep-T"


@dataclass(frozen=True)
class Cohort:
    metadata: str
    repertoires: str
    description: str


COHORTS: dict[str, Cohort] = {
    "malid": Cohort(
        "Mal-ID/metadata.tsv",
        "Mal-ID/repertoires",
        "Zaslavsky/Mal-ID (HIV, lupus, influenza, COVID-19, and T1D)",
    ),
    "mitchell-t1d": Cohort(
        "immunoSEQ/Mitchell_T1D/metadata.tsv",
        "immunoSEQ/Mitchell_T1D/repertoires",
        "Mitchell T1D",
    ),
    "rawat-t1d": Cohort(
        "immunoSEQ/Rawat_T1D/metadata.tsv",
        "immunoSEQ/Rawat_T1D/repertoires",
        "Rawat T1D",
    ),
    "tb": Cohort(
        "immunoSEQ/Musvosvi_TB/metadata.tsv",
        "immunoSEQ/Musvosvi_TB/repertoires",
        "Musvosvi tuberculosis progression",
    ),
    "ra": Cohort(
        "immunoSEQ/Savola_RA/metadata.tsv",
        "immunoSEQ/Savola_RA/repertoires",
        "Savola rheumatoid arthritis",
    ),
    "cmv": Cohort(
        "immunoSEQ/Emerson_CMV/metadata.tsv",
        "immunoSEQ/Emerson_CMV/repertoires",
        "Emerson CMV serostatus",
    ),
}

TASKS = ("disease", "drivers", "depth", "demographics")
DRIVER_MATCHES = "Mal-ID/vdjdb_minervina_driver_seq_matches.csv"
DEPTH_INDICES = ("Mal-ID/scaling_exp_depth_indices_max75k.json.gz",)
REQUIRED_METADATA_COLUMNS = {
    "participant_label",
    "specimen_label",
    "disease",
}


def selected_cohorts(tasks: Sequence[str], cohorts: Sequence[str]) -> tuple[str, ...]:
    """Resolve explicit cohorts or the cohorts implied by the selected tasks."""
    if cohorts:
        selected = list(cohorts)
        # These published auxiliary assets are keyed to Mal-ID repertoire IDs.
        if {"drivers", "depth"}.intersection(tasks) and "malid" not in selected:
            selected.append("malid")
        return tuple(dict.fromkeys(selected))
    # Driver labels and the published nested depth indices refer to Mal-ID.
    if set(tasks).issubset({"drivers", "depth"}):
        return ("malid",)
    return tuple(COHORTS)


def local_patterns(tasks: Sequence[str], cohorts: Sequence[str]) -> tuple[str, ...]:
    """Return repository-relative files needed for a task/cohort selection."""
    patterns: list[str] = []
    for name in selected_cohorts(tasks, cohorts):
        cohort = COHORTS[name]
        patterns.extend((cohort.metadata, f"{cohort.repertoires}/**"))

    if "drivers" in tasks:
        patterns.append(DRIVER_MATCHES)
    if "depth" in tasks:
        patterns.extend(DEPTH_INDICES)
    return tuple(dict.fromkeys(patterns))


def detect_source_prefix(repo_files: Iterable[str]) -> str:
    """Detect whether the repository stores its dataset at root or in data/."""
    names = set(repo_files)
    marker = COHORTS["malid"].metadata
    if marker in names:
        return ""
    if f"data/{marker}" in names:
        return "data"
    raise RuntimeError(
        "Could not find the BenchRep-T layout in the dataset repository. "
        f"Expected either {marker!r} or {'data/' + marker!r}. "
        "Use --source-prefix if the repository layout has changed."
    )


def _remote_patterns(patterns: Sequence[str], source_prefix: str) -> list[str]:
    prefix = source_prefix.strip("/")
    return [f"{prefix}/{pattern}" if prefix else pattern for pattern in patterns]


def _snapshot_local_dir(output_dir: Path, source_prefix: str) -> Path:
    """Choose local_dir while preserving the evaluator-facing output layout."""
    prefix = source_prefix.strip("/")
    if not prefix:
        return output_dir
    if "/" in prefix or output_dir.name != prefix:
        raise ValueError(
            f"Remote files are below {prefix!r}; output directory must therefore "
            f"end in /{prefix} so paths can be materialized without a second copy."
        )
    return output_dir.parent


def _metadata_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle, delimiter="\t"), None)
    return set(row or [])


def validate_data(
    output_dir: Path, tasks: Sequence[str], cohorts: Sequence[str]
) -> list[str]:
    """Validate the materialized file contract and return human-readable results."""
    errors: list[str] = []
    results: list[str] = []
    for name in selected_cohorts(tasks, cohorts):
        cohort = COHORTS[name]
        metadata = output_dir / cohort.metadata
        repertoire_dir = output_dir / cohort.repertoires
        if not metadata.is_file():
            errors.append(f"missing metadata: {metadata}")
        else:
            missing_columns = REQUIRED_METADATA_COLUMNS - _metadata_columns(metadata)
            if missing_columns:
                errors.append(
                    f"{metadata} lacks columns: {', '.join(sorted(missing_columns))}"
                )
        if not repertoire_dir.is_dir() or not any(repertoire_dir.glob("*.tsv*")):
            errors.append(f"no repertoire TSV files found in: {repertoire_dir}")
        results.append(
            f"{name}: metadata={metadata}, repertoires={repertoire_dir}"
        )

    if "drivers" in tasks and not (output_dir / DRIVER_MATCHES).is_file():
        errors.append(f"missing driver matches: {output_dir / DRIVER_MATCHES}")
    if "depth" in tasks and not any(
        (output_dir / relative_path).is_file() for relative_path in DEPTH_INDICES
    ):
        errors.append(
            "missing depth indices: expected at least one of "
            + ", ".join(str(output_dir / path) for path in DEPTH_INDICES)
        )

    if errors:
        raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return results


def download_data(
    *,
    repo_id: str,
    output_dir: Path,
    tasks: Sequence[str],
    cohorts: Sequence[str],
    revision: str | None = None,
    source_prefix: str = "auto",
    token: str | bool | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Download selected private dataset files and validate their local layout."""
    patterns = local_patterns(tasks, cohorts)
    if dry_run:
        return list(patterns)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install the project again or run "
            "`python -m pip install huggingface_hub`."
        ) from exc

    resolved_prefix = source_prefix
    if resolved_prefix == "auto":
        repo_files = HfApi().list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        resolved_prefix = detect_source_prefix(repo_files)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns=_remote_patterns(patterns, resolved_prefix),
        local_dir=_snapshot_local_dir(output_dir, resolved_prefix),
    )
    return validate_data(output_dir, tasks, cohorts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download private BenchRep-T files in their native Mal-ID/ and "
            "immunoSEQ/ layout for disease, driver, depth, and demographic tasks."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Evaluator-facing data directory (default: ./data).",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        action="append",
        dest="tasks",
        help="Task to fetch; repeat for multiple tasks (default: all tasks).",
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(COHORTS),
        action="append",
        dest="cohorts",
        help="Cohort to fetch; repeat as needed (default: all relevant cohorts).",
    )
    parser.add_argument(
        "--revision",
        help="Dataset branch, tag, or commit (default: repository main revision).",
    )
    parser.add_argument(
        "--source-prefix",
        default="auto",
        help="Remote data prefix: auto, an empty string, or data (default: auto).",
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help=(
            "Read the access token from this environment variable. By default "
            "huggingface_hub uses HF_TOKEN or the token saved by `hf auth login`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected repository-relative paths without network access.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing local data without contacting Hugging Face.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = tuple(dict.fromkeys(args.tasks or TASKS))
    cohorts = tuple(dict.fromkeys(args.cohorts or ()))

    try:
        if args.validate_only:
            lines = validate_data(args.output_dir.resolve(), tasks, cohorts)
        else:
            token: str | bool | None = None
            if args.token_env:
                token = os.environ.get(args.token_env)
                if not token:
                    raise RuntimeError(
                        f"Environment variable {args.token_env!r} is not set."
                    )
            lines = download_data(
                repo_id=args.repo_id,
                output_dir=args.output_dir,
                tasks=tasks,
                cohorts=cohorts,
                revision=args.revision,
                source_prefix=args.source_prefix,
                token=token,
                dry_run=args.dry_run,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if "401" in str(exc) or "gated" in str(exc).lower():
            print(
                "Authenticate with `hf auth login` or set HF_TOKEN to a token "
                "that has access to the private dataset.",
                file=sys.stderr,
            )
        return 1

    if args.dry_run:
        print("Selected dataset paths:")
    else:
        print("BenchRep-T data are ready:")
    for line in lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
