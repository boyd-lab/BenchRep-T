"""
Reformat the external T1D metadata file to use MAL-ID column conventions
so the same evaluator code paths work across both cohorts.

Input  (Adaptive-style, e.g. data/metadata_T1D.tsv pre-conversion):
    sample_name, age_at_diagnosis, sex, race, ethnicity,
    t1d_duration_days, disease_label, fold

Output (MAL-ID style — matches data/metadata_malid.tsv):
    participant_label, specimen_label, disease, specimen_time_point,
    study_name, available_gene_loci, disease_subtype, age, sex, ancestry,
    malid_cross_validation_fold_id_when_in_test_set

This script is idempotent: if it sees a file already in MAL-ID style it will
re-emit it unchanged.

Usage:
    python external_data_process/harmonize_t1d_metadata.py \
        --metadata_path data/metadata_T1D.tsv
"""

import argparse
import os

import pandas as pd


MALID_COLUMNS = [
    'participant_label',
    'specimen_label',
    'disease',
    'specimen_time_point',
    'study_name',
    'available_gene_loci',
    'disease_subtype',
    'age',
    'sex',
    'ancestry',
    'malid_cross_validation_fold_id_when_in_test_set',
]

DISEASE_MAP = {'Healthy': 'Healthy/Background', 'T1D': 'T1D'}
SEX_MAP = {'Female': 'F', 'Male': 'M'}


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Map a single Adaptive-style external T1D metadata DataFrame to MAL-ID style."""
    if set(MALID_COLUMNS).issubset(df.columns):
        return df[MALID_COLUMNS].copy()

    out = pd.DataFrame()
    out['participant_label'] = df['sample_name']
    out['specimen_label'] = df['sample_name']
    out['disease'] = df['disease_label'].map(DISEASE_MAP)
    if out['disease'].isna().any():
        bad = df.loc[out['disease'].isna(), 'disease_label'].unique()
        raise ValueError(f"Unmapped disease_label values: {bad}")
    out['specimen_time_point'] = ''
    out['study_name'] = 'external_T1D'
    out['available_gene_loci'] = 'GeneLocus.TCR'
    out['disease_subtype'] = out['disease'].map(
        lambda v: 'T1D - external' if v == 'T1D' else ''
    )
    out['age'] = df['age_at_diagnosis']
    out['sex'] = df['sex'].map(SEX_MAP).fillna('')
    # External T1D cohort is treated as a single-ancestry group ("White") for
    # downstream demographic-adjusted analyses, regardless of the heterogeneous
    # race / ethnicity values in the source file.
    out['ancestry'] = 'White'
    out['malid_cross_validation_fold_id_when_in_test_set'] = df['fold']

    return out[MALID_COLUMNS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata_path', required=True,
                   help='Path to external T1D metadata TSV (will be overwritten in place)')
    p.add_argument('--output', default=None,
                   help='Optional explicit output path (default: overwrite --metadata_path)')
    args = p.parse_args()

    df = pd.read_csv(args.metadata_path, sep='\t')
    out = harmonize(df)

    output_path = args.output or args.metadata_path
    out.to_csv(output_path, sep='\t', index=False)

    print(f"Wrote {len(out)} rows ({out.shape[1]} columns) to {output_path}")
    print("Disease counts:")
    print(out['disease'].value_counts().to_string())
    print("\nFold counts:")
    print(out['malid_cross_validation_fold_id_when_in_test_set'].value_counts()
          .sort_index().to_string())


if __name__ == '__main__':
    main()
