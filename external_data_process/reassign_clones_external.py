"""
Reassign clone_id in external_data_airr repertoire files using bcr_clones.assign_clones.

For each *.tsv file directly under
/backups/chihoim/datasets/external_data_airr/data_cleaned/:
  1. Run bcr_clones.assign_clones on (v_call, j_call, cdr3_aa) to produce a new
     `clone_id` column. Any existing `clone_id` is overwritten; `cloneResolved`
     is left untouched.
  2. Overwrite the file in place (atomic rename via .tmp).

Usage:
    python external_data_process/reassign_clones_external.py \
        --root /backups/chihoim/datasets/external_data_airr/data_cleaned \
        --workers 16
"""

import argparse
import os
import sys
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '/users/chihoim/software/bcr_clones')
import bcr_clones  # noqa: E402


V_COL = 'v_call'
J_COL = 'j_call'
V_NO_ALLELE_COL = 'v_gene_no_allele'
J_NO_ALLELE_COL = 'j_gene_no_allele'
CDR3_COL = 'cdr3_aa'
CLONE_COL = 'clone_id'
IDENTITY_THRESHOLD = 0.95  # overridden in main() from --identity_threshold
LINKAGE_METHOD = 'single'


def process_file(path_str):
    path = Path(path_str)
    try:
        df = pd.read_csv(path, sep='\t', low_memory=False)

        for required in (V_COL, J_COL, CDR3_COL):
            if required not in df.columns:
                return (path_str, False, f"missing required column {required}")

        if CLONE_COL in df.columns:
            df = df.drop(columns=[CLONE_COL])

        df[V_NO_ALLELE_COL] = df[V_COL].str.split('*').str[0]
        df[J_NO_ALLELE_COL] = df[J_COL].str.split('*').str[0]

        df = bcr_clones.assign_clones(
            df,
            heavy_v_gene_col=V_NO_ALLELE_COL,
            heavy_j_gene_col=J_NO_ALLELE_COL,
            heavy_cdr3_col=CDR3_COL,
            identity_threshold=IDENTITY_THRESHOLD,
            linkage_method=LINKAGE_METHOD,
            heavy_clone_output_col_name=CLONE_COL,
            verbose=False,
        )

        df = df.drop(columns=[V_NO_ALLELE_COL, J_NO_ALLELE_COL])

        tmp_path = path.with_suffix(path.suffix + '.tmp')
        df.to_csv(tmp_path, sep='\t', index=False)
        os.replace(tmp_path, path)
        return (path_str, True, None)
    except Exception as e:
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return (path_str, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def enumerate_files(root, file_glob):
    root = Path(root)
    return sorted(str(p) for p in root.glob(file_glob) if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/backups/chihoim/datasets/external_data_airr/data_cleaned')
    ap.add_argument('--file_glob', default='*.tsv',
                    help='Glob pattern (relative to --root) matching repertoire files (default *.tsv)')
    ap.add_argument('--identity_threshold', type=float, default=0.95,
                    help='Fraction identity required to be in the same clone (default 0.95)')
    ap.add_argument('--workers', type=int, default=min(max(cpu_count() - 2, 1), 32))
    ap.add_argument('--limit', type=int, default=None, help='Process only the first N files (for dry runs)')
    args = ap.parse_args()

    global IDENTITY_THRESHOLD
    IDENTITY_THRESHOLD = args.identity_threshold
    print(f"Identity threshold: {IDENTITY_THRESHOLD}")

    files = enumerate_files(args.root, args.file_glob)
    print(f"Root: {args.root}")
    print(f"Glob: {args.file_glob}")
    print(f"Total files: {len(files)}")

    if args.limit is not None:
        files = files[:args.limit]
        print(f"Limiting to first {len(files)} files")

    print(f"Using {args.workers} workers")

    failures = []
    successes = 0
    with Pool(processes=args.workers) as pool:
        for path_str, ok, err in tqdm(pool.imap_unordered(process_file, files),
                                       total=len(files), desc='Reassigning clones'):
            if ok:
                successes += 1
            else:
                failures.append((path_str, err))

    print(f"\nDone. Success: {successes}/{len(files)}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for p, err in failures[:20]:
            print(f"  {p}\n    {err.splitlines()[0] if err else ''}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == '__main__':
    main()
