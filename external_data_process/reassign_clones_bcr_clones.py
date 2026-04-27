"""
Reassign clone_id in malid_tcr_airr repertoire files using bcr_clones.assign_clones.

For each part_table_*.tsv.gz file under
/backups/chihoim/datasets/malid_tcr_airr/data_cleaned_per_participant/rep0*/:
  1. Rename original `clone_id` -> `clone_id_old` (idempotent: skipped if already renamed).
  2. Run bcr_clones.assign_clones on (v_call, j_call, cdr3_aa) to produce a new `clone_id`.
  3. Overwrite the file in place (atomic rename via .tmp.gz).

Usage:
    python external_data_process/reassign_clones_bcr_clones.py \
        --root /backups/chihoim/datasets/malid_tcr_airr/data_cleaned_per_participant \
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
CLONE_OLD_COL = 'clone_id_old'
IDENTITY_THRESHOLD = 0.90  # overridden in main() from --identity_threshold
LINKAGE_METHOD = 'single'


def process_file(path_str):
    path = Path(path_str)
    try:
        df = pd.read_csv(path, sep='\t', compression='gzip', low_memory=False)

        if CLONE_OLD_COL not in df.columns:
            if CLONE_COL not in df.columns:
                return (path_str, False, f"missing both {CLONE_COL} and {CLONE_OLD_COL}")
            df = df.rename(columns={CLONE_COL: CLONE_OLD_COL})
        else:
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
        df.to_csv(tmp_path, sep='\t', index=False, compression='gzip')
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


def enumerate_files(root, subdir_glob):
    root = Path(root)
    subdirs = sorted(p for p in root.glob(subdir_glob) if p.is_dir())
    files = []
    for sd in subdirs:
        files.extend(sorted(sd.glob('part_table_*.tsv.gz')))
    return [str(p) for p in files], subdirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/backups/chihoim/datasets/malid_tcr_airr/data_cleaned_per_participant')
    ap.add_argument('--subdir_glob', default='rep0*', help='Glob pattern matching target subdirectories')
    ap.add_argument('--identity_threshold', type=float, default=0.90,
                    help='Fraction identity required to be in the same clone (default 0.90)')
    ap.add_argument('--workers', type=int, default=min(max(cpu_count() - 2, 1), 32))
    ap.add_argument('--limit', type=int, default=None, help='Process only the first N files (for dry runs)')
    args = ap.parse_args()

    global IDENTITY_THRESHOLD
    IDENTITY_THRESHOLD = args.identity_threshold
    print(f"Identity threshold: {IDENTITY_THRESHOLD}")

    files, subdirs = enumerate_files(args.root, args.subdir_glob)
    print(f"Found {len(subdirs)} {args.subdir_glob!r} subdirectories:")
    for sd in subdirs:
        n = len(list(sd.glob('part_table_*.tsv.gz')))
        print(f"  {sd.name}: {n} files")
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
