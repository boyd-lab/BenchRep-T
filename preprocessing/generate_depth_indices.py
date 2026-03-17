#!/usr/bin/env python3
"""
Generate subsampling indices for sequencing depth scaling law experiments.

For each repertoire with >= 100,000 sequences, generates N_REPS independent
random samples of 100,000 row indices (without replacement). Depth K uses
the first K indices of each sample, so smaller depths are always nested
subsets of larger ones — removing one source of cross-depth variance.

Depths: 1000, 5000, 10000, 25000, 50000, 100000
Repetitions: 5

Output JSON structure:
    {
        "<rep_id>": {
            "0": [idx0, idx1, ..., idx99999],   # repetition 0
            "1": [...],                          # repetition 1
            ...
            "4": [...]                           # repetition 4
        },
        ...
    }

To get indices for rep_id, depth D, repetition R:
    indices = data[rep_id][str(R)][:D]

Per-(repertoire, repetition) RNGs are derived from a master seed via
numpy SeedSequence, ensuring independence across both dimensions.
The master seed is embedded in the output filename for traceability.

Usage:
    python generate_depth_indices.py <repertoire_dir> <output_path> [--seed SEED]

Example:
    python generate_depth_indices.py /path/to/repertoires data/depth_indices_seed42.json.gz
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEPTHS = [1000, 5000, 10000, 25000, 50000, 100_000]
MIN_SEQUENCES = 100_000
N_REPS = 5
DEFAULT_SEED = 42


def count_sequences(path: Path) -> int:
    """Count data rows (excluding header) in a gzipped TSV."""
    with gzip.open(path, "rt") as f:
        return sum(1 for _ in f) - 1  # subtract header row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repertoire_dir",
        type=Path,
        help="Directory containing part_table_*.tsv.gz files",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the indices file (.json or .json.gz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if not args.repertoire_dir.is_dir():
        print(f"ERROR: {args.repertoire_dir} is not a directory", file=sys.stderr)
        return 1

    files = sorted(args.repertoire_dir.glob("part_table_*.tsv.gz"))
    if not files:
        print(
            f"ERROR: No part_table_*.tsv.gz files found in {args.repertoire_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(files)} repertoire files.")
    print(f"Filtering to those with >= {MIN_SEQUENCES:,} sequences...\n")

    qualifying: list[tuple[Path, int]] = []
    skipped = 0
    for path in tqdm(files, unit="file"):
        n = count_sequences(path)
        if n >= MIN_SEQUENCES:
            qualifying.append((path, n))
        else:
            skipped += 1

    print(f"\nQualifying (>= {MIN_SEQUENCES:,} sequences): {len(qualifying)}")
    print(f"Skipped (too small):                        {skipped}")

    if not qualifying:
        print("ERROR: No qualifying repertoires found.", file=sys.stderr)
        return 1

    # Derive independent per-(repertoire, repetition) RNGs from the master seed.
    # SeedSequence guarantees statistical independence between child streams.
    ss = np.random.SeedSequence(args.seed)
    n_qual = len(qualifying)
    child_seeds = ss.spawn(n_qual * N_REPS)

    print(f"\nGenerating {N_REPS} permutations x {n_qual} repertoires (master seed={args.seed})...\n")

    result: dict[str, dict[str, list[int]]] = {}
    for i, (path, n_seq) in enumerate(tqdm(qualifying, unit="repertoire")):
        # rep_id: filename without .tsv.gz extension
        rep_id = path.name[: -len(".tsv.gz")]
        rep_data: dict[str, list[int]] = {}
        for r in range(N_REPS):
            rng = np.random.default_rng(child_seeds[i * N_REPS + r])
            # Sample MIN_SEQUENCES indices without replacement.
            # Depth D uses the first D elements (nesting property).
            indices = rng.choice(n_seq, size=MIN_SEQUENCES, replace=False)
            rep_data[str(r)] = indices.tolist()
        result[rep_id] = rep_data

    # Write output (auto-detect gzip from extension)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {args.output} ...")
    if args.output.name.endswith(".json.gz"):
        with gzip.open(args.output, "wt", encoding="utf-8") as f:
            json.dump(result, f)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"\nDone.")
    print(f"  Repertoires written : {len(result)}")
    print(f"  Repetitions each    : {N_REPS}")
    print(f"  Depths available    : {DEPTHS}")
    print(f"  Master seed         : {args.seed}")
    print(f"\nUsage: data[rep_id][str(repetition)][:depth]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
