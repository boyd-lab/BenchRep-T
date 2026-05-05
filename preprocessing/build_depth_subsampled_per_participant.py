#!/usr/bin/env python3
"""

****For Mal-ID's use****

Build depth-subsampled per-participant repertoire files from per-specimen
repertoires and a precomputed depth-indices file.

For each (rep, depth) combination in the indices file, this script takes
the first `depth` indices of each specimen's `rep` permutation, slices the
specimen's TSV by those rows, and stacks the slices across all specimens
that belong to the same participant.

Output layout:
    <output-dir>/rep{R}_n{D}/part_table_{participant}.tsv.gz

For a participant with K specimens present in the indices file, the output
at depth D contains K * D rows (per-specimen depth, stacked).

Specimens absent from the indices file (had < min_sequences) are skipped.
Participants with no qualifying specimen produce no output.

Usage:
    python preprocessing/build_depth_subsampled_per_participant.py \\
        --input-dir  <path/to/data_per_specimen> \\
        --output-dir <path/to/data_per_participant> \\
        --indices    data/depth_indices_max75k.json.gz \\
        --workers    8
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_participant(specimen_key: str) -> str:
    """`part_table_BFI-0000234_M124-S014` -> `BFI-0000234`."""
    stem = specimen_key.removeprefix("part_table_")
    return stem.rsplit("_", 1)[0]


def process_participant(
    participant: str,
    specimen_keys: list[str],
    indices_for_participant: dict[str, dict[str, list[int]]],
    depths: list[int],
    n_reps: int,
    input_dir: Path,
    output_dir: Path,
) -> tuple[str, int]:
    specimen_dfs: dict[str, pd.DataFrame] = {}
    for key in specimen_keys:
        path = input_dir / f"{key}.tsv.gz"
        specimen_dfs[key] = pd.read_csv(path, sep="\t", low_memory=False)

    n_written = 0
    for r in range(n_reps):
        rep_subsets: dict[str, pd.DataFrame] = {}
        for key, df in specimen_dfs.items():
            indices = indices_for_participant[key][str(r)]
            rep_subsets[key] = df.iloc[indices].reset_index(drop=True)

        for d in depths:
            parts = [rep_subsets[k].iloc[:d] for k in specimen_keys]
            stacked = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            out_path = output_dir / f"rep{r}_n{d}" / f"part_table_{participant}.tsv.gz"
            with gzip.open(out_path, "wt") as f:
                stacked.to_csv(f, sep="\t", index=False)
            n_written += 1

    return participant, n_written


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-specimen repertoire TSVs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for depth-subsampled per-participant files.",
    )
    p.add_argument(
        "--indices",
        type=Path,
        default=Path("data/depth_indices_max75k.json.gz"),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel processes across participants (default: 1)",
    )
    args = p.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not args.indices.is_file():
        print(f"ERROR: indices file not found: {args.indices}", file=sys.stderr)
        return 1

    print(f"Loading indices: {args.indices}")
    opener = gzip.open if args.indices.name.endswith(".gz") else open
    with opener(args.indices, "rt") as f:
        idx_data = json.load(f)

    depths: list[int] = idx_data["depths"]
    n_reps: int = idx_data["n_reps"]
    repertoires: dict[str, dict[str, list[int]]] = idx_data["repertoires"]

    by_participant: dict[str, list[str]] = defaultdict(list)
    for key in repertoires:
        by_participant[parse_participant(key)].append(key)
    by_participant = {p: sorted(v) for p, v in sorted(by_participant.items())}

    print(f"  specimens : {len(repertoires)}")
    print(f"  participants: {len(by_participant)}")
    print(f"  depths    : {depths}")
    print(f"  n_reps    : {n_reps}")
    print(f"Input dir : {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Workers   : {args.workers}")

    missing = [
        f"{k}.tsv.gz"
        for k in repertoires
        if not (args.input_dir / f"{k}.tsv.gz").is_file()
    ]
    if missing:
        print(
            f"ERROR: {len(missing)} specimen file(s) referenced by indices are missing in {args.input_dir}:",
            file=sys.stderr,
        )
        for name in missing[:10]:
            print(f"  {name}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        return 1

    for r in range(n_reps):
        for d in depths:
            (args.output_dir / f"rep{r}_n{d}").mkdir(parents=True, exist_ok=True)

    total_expected = len(by_participant) * n_reps * len(depths)
    print(f"\nWill produce {total_expected} output files "
          f"({len(by_participant)} participants x {n_reps} reps x {len(depths)} depths)\n")

    total_written = 0
    if args.workers <= 1:
        for participant, keys in tqdm(by_participant.items(), unit="participant"):
            sub_idx = {k: repertoires[k] for k in keys}
            _, n = process_participant(
                participant, keys, sub_idx, depths, n_reps,
                args.input_dir, args.output_dir,
            )
            total_written += n
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for participant, keys in by_participant.items():
                sub_idx = {k: repertoires[k] for k in keys}
                futures.append(ex.submit(
                    process_participant,
                    participant, keys, sub_idx, depths, n_reps,
                    args.input_dir, args.output_dir,
                ))
            for fut in tqdm(as_completed(futures), total=len(futures), unit="participant"):
                _, n = fut.result()
                total_written += n

    print(f"\nDone. Wrote {total_written} files (expected {total_expected}).")
    return 0 if total_written == total_expected else 1


if __name__ == "__main__":
    sys.exit(main())
