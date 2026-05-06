"""
Compute random-chance recall@k baseline per (disease, k) for the driver
sequence identification plot.

Two baselines are reported per (disease, k):

  * random_recall_at_k        analytical, mean(min(k/N, 1.0)) over
                              disease-positive repertoires (legacy).
  * random_recall_at_k_mc     Monte Carlo. For each repertoire, simulate
                              random orderings of its N unique CDR3s,
                              count how many ground-truth drivers fall in
                              the top k, divide by |drivers_csv|, average
                              over trials, then average over repertoires.
                              Denominator matches the evaluator (driver
                              count from the driver-seqs CSV, not the
                              intersection with the repertoire).
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def unique_cdr3_set(file_path, sequence_col='cdr3_aa'):
    df = pd.read_csv(file_path, sep='\t', usecols=[sequence_col])
    return set(df[sequence_col].dropna().unique())


def load_drivers_by_file(driver_seqs_path, disease):
    df = pd.read_csv(driver_seqs_path)
    df = df[df['disease'] == disease]
    return {fn: set(g['sample_cdr3'].unique())
            for fn, g in df.groupby('filename')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata_path', default='data/metadata_malid.tsv')
    ap.add_argument('--repertoire_dir', required=True,
                    help='Directory containing per-specimen repertoire TSVs '
                         '(e.g. data/malid_clean/data_per_specimen).')
    ap.add_argument('--driver_seqs_path',
                    default='data/vdjdb_matches_expanded.csv',
                    help='Ground-truth driver CDR3s CSV (columns: disease, '
                         'sample_cdr3, filename, ...).')
    ap.add_argument('--diseases', nargs='+',
                    default=['HIV', 'Covid19', 'Influenza'])
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[100, 1000, 10000])
    ap.add_argument('--n_trials', type=int, default=10000,
                    help='Number of random orderings per repertoire.')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--output_csv', default='random_baseline_recall.csv')
    args = ap.parse_args()

    metadata = pd.read_csv(args.metadata_path, sep='\t')
    rng = np.random.default_rng(args.seed)

    rows = []
    for disease in args.diseases:
        disease_meta = metadata[metadata['disease'] == disease].copy()
        disease_meta['filename'] = (
            'part_table_' + disease_meta['participant_label'].astype(str)
            + '_' + disease_meta['specimen_label'].astype(str)
        )
        disease_meta['file_path'] = disease_meta['filename'].apply(
            lambda f: os.path.join(args.repertoire_dir, f + '.tsv.gz')
        )

        drivers_by_file = load_drivers_by_file(args.driver_seqs_path, disease)

        eligible = disease_meta[
            disease_meta['file_path'].apply(os.path.exists)
            & disease_meta['filename'].isin(drivers_by_file)
        ]
        print(f"\n=== {disease}: {len(eligible)} disease-positive "
              f"repertoires with ground-truth drivers ===")

        # per-repertoire: N (unique CDR3s), M (drivers ∩ repertoire),
        # D_csv (drivers from CSV, full denominator), and per-k mean recall.
        per_rep = []
        for _, row in tqdm(eligible.iterrows(), total=len(eligible),
                           desc=disease):
            try:
                rep_set = unique_cdr3_set(row['file_path'])
            except Exception as e:
                print(f"  skip {row['filename']}: {e}")
                continue

            N = len(rep_set)
            drivers_csv = drivers_by_file[row['filename']]
            D_csv = len(drivers_csv)
            if N == 0 or D_csv == 0:
                continue
            M = len(drivers_csv & rep_set)

            entry = {'N': N, 'M': M, 'D_csv': D_csv}
            for k in args.ks:
                draw_k = min(k, N)
                # Hypergeometric: # of "good" items (drivers present in
                # repertoire) in a uniform random subset of size draw_k
                # — equivalent to top-k of a random shuffle.
                if M == 0:
                    hits = np.zeros(args.n_trials, dtype=np.int64)
                else:
                    hits = rng.hypergeometric(M, N - M, draw_k,
                                              size=args.n_trials)
                entry[f'recall_at_{k}'] = (hits / D_csv).mean()
            per_rep.append(entry)

        if not per_rep:
            print(f"  no eligible repertoires for {disease}")
            continue

        ns = np.array([e['N'] for e in per_rep], dtype=float)
        ms = np.array([e['M'] for e in per_rep], dtype=float)
        ds = np.array([e['D_csv'] for e in per_rep], dtype=float)
        print(f"  N (unique CDR3s): mean={ns.mean():.0f}, "
              f"median={np.median(ns):.0f}")
        print(f"  drivers in CSV per rep: mean={ds.mean():.1f}, "
              f"of which present in rep: mean={ms.mean():.1f}")

        for k in args.ks:
            analytical = np.minimum(k / ns, 1.0).mean()
            mc_per_rep = np.array([e[f'recall_at_{k}'] for e in per_rep])
            mc_recall = mc_per_rep.mean()
            print(f"  Random recall@{k}: analytical={analytical:.4f}, "
                  f"MC={mc_recall:.4f}")
            rows.append({
                'disease': disease,
                'k': k,
                'n_repertoires': len(per_rep),
                'mean_unique_cdr3s': float(ns.mean()),
                'mean_drivers_in_csv': float(ds.mean()),
                'mean_drivers_in_repertoire': float(ms.mean()),
                'random_recall_at_k': float(analytical),
                'random_recall_at_k_mc': float(mc_recall),
                'n_trials': args.n_trials,
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
