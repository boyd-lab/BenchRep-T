"""Dump sample lists for demographic-adjusted cohort experiments.

For each disease in {HIV, Lupus, Influenza, Covid19}, write one TSV per
cohort (1 demographic-adjusted + 5 random baselines = 6 per disease,
24 total) listing the samples that go into that cohort.
"""

import argparse
import os

import pandas as pd

from evals.emerson_2017_disease_classification import Emerson2017Evaluator


DISEASES = ['HIV', 'Lupus', 'Influenza', 'Covid19']
RANDOM_BASELINE_SEEDS = [7, 14, 21, 28, 35]


def build_sample_name(df):
    return df['participant_label'].astype(str) + '_' + df['specimen_label'].astype(str)


def dump_cohort(df, out_path, target_disease, cohort_kind, seed=None):
    out = df.copy()
    out['sample_name'] = build_sample_name(out)
    cols = [
        'sample_name', 'participant_label', 'specimen_label',
        'disease', 'label', 'age', 'sex', 'ancestry',
        'malid_cross_validation_fold_id_when_in_test_set',
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values('sample_name').reset_index(drop=True)
    out.to_csv(out_path, sep='\t', index=False)
    seed_str = f", seed={seed}" if seed is not None else ""
    n_disease = int((out['label'] == 1).sum())
    n_healthy = int((out['label'] == 0).sum())
    print(f"  -> {out_path}  ({cohort_kind}{seed_str}): "
          f"disease={n_disease}, healthy={n_healthy}, total={len(out)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str,
                        default='data/metadata_malid.tsv')
    parser.add_argument('--out_dir', type=str,
                        default='demographic_cohort_samples')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    evaluator = Emerson2017Evaluator()
    metadata = evaluator.load_metadata(args.metadata_path)

    for disease in DISEASES:
        print(f"\n=== {disease} ===")

        # 1. demographic-adjusted cohort (single matched run)
        adjusted = evaluator.prepare_disease_data(
            metadata, target_disease=disease,
            adjust_distribution_by_demographics=True,
            random_baseline=False,
        )
        out_path = os.path.join(
            args.out_dir,
            f"{disease}_demographic_adjusted.tsv",
        )
        dump_cohort(adjusted, out_path, disease, 'demographic_adjusted')

        # 2. five random baselines
        for seed in RANDOM_BASELINE_SEEDS:
            baseline = evaluator.prepare_disease_data(
                metadata, target_disease=disease,
                adjust_distribution_by_demographics=True,
                random_baseline=True,
                random_baseline_seed=seed,
            )
            out_path = os.path.join(
                args.out_dir,
                f"{disease}_random_baseline_seed{seed}.tsv",
            )
            dump_cohort(baseline, out_path, disease,
                        'random_baseline', seed=seed)


if __name__ == '__main__':
    main()
