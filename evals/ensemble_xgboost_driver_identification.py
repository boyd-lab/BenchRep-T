"""
Driver sequence identification for the Ensemble XGBoost model.

Loads per-fold models saved by ensemble_xgboost_disease_classification.py
(--model_save_dir) and scores each CDR3 in the test repertoires using the
trained XGBoost models directly on per-sequence feature vectors.

Mirrors ensemble_regression_driver_identification.py but uses XGBoost
probabilities instead of logistic-regression decision-function scores.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.ensemble_xgboost import XGBoostKmer
from utils.repertoire_io import load_raw_repertoire


class EnsembleXGBoostDriverEvaluator:

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, model_save_dir, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call'):
        self.model_save_dir = model_save_dir
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)
        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}': "
              f"{n_disease} disease, {n_healthy} healthy, {len(filtered)} total")
        return filtered

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: os.path.join(
                data_dir,
                f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}"
            ), axis=1
        )
        return metadata

    def filter_existing_files(self, metadata):
        original = len(metadata)
        metadata = metadata.copy()
        metadata['file_exists'] = metadata['file_path'].apply(os.path.exists)
        filtered = metadata[metadata['file_exists']].drop(columns=['file_exists'])
        missing = original - len(filtered)
        if missing > 0:
            print(f"Note: {missing}/{original} files not found. "
                  f"Proceeding with {len(filtered)}.")
        return filtered

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def load_driver_sequences(self, driver_seqs_path, target_disease):
        """
        Load ground truth driver CDR3s grouped by repertoire filename stem.
        Returns dict: {filename_stem -> set of CDR3 strings}
        """
        df = pd.read_csv(driver_seqs_path)
        disease_df = df[df['disease'] == target_disease]
        drivers_by_file = {}
        for filename, group in disease_df.groupby('filename'):
            drivers_by_file[filename] = set(group['sample_cdr3'].unique())
        total = sum(len(v) for v in drivers_by_file.values())
        print(f"Ground truth for '{target_disease}': "
              f"{len(drivers_by_file)} repertoires, {total} driver CDR3s")
        return drivers_by_file

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              driver_seqs_path, ks,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3,
                              allowed_participants=None,
                              max_repertoires=None,
                              output_csv=None,
                              ext_metadata_path=None, ext_data_dir=None,
                              ext_file_template='{participant_label}_TCRB.tsv'):
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if ext_metadata_path is not None:
            from utils.cohort_merge import prepare_merged_cohort
            metadata = prepare_merged_cohort(
                metadata, ext_metadata_path, ext_data_dir, target_disease,
                ext_file_template=ext_file_template,
                healthy_label=self.HEALTHY_LABEL,
                fold_col=fold_col, disease_col=disease_col,
            )

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)}/{before} specimens "
                  f"via allowed_participants.")

        if max_repertoires is not None and len(metadata) > max_repertoires:
            metadata = metadata.sample(n=max_repertoires, random_state=7)
            print(f"Subsampled to {max_repertoires} repertoires for debug run.")

        drivers_by_file = self.load_driver_sequences(driver_seqs_path, target_disease)

        all_results = []
        fold_summaries = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            fold_dir = os.path.join(self.model_save_dir, target_disease,
                                    f'fold{test_fold}')
            if not os.path.isdir(fold_dir):
                print(f"  WARNING: no saved model at {fold_dir}; skipping fold.")
                fold_summaries.append({
                    'fold': test_fold, 'n_repertoires': 0,
                    **{f'mean_precision_at_{k}': float('nan') for k in ks},
                    **{f'mean_recall_at_{k}': float('nan') for k in ks},
                })
                continue

            print(f"  Loading model from {fold_dir} ...")
            model = XGBoostKmer.load(fold_dir)

            test_data = metadata[metadata[fold_col] == test_fold]
            print(f"  Test specimens: {len(test_data)}")

            fold_precisions = {k: [] for k in ks}
            fold_recalls = {k: [] for k in ks}

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Driver scoring"):
                fp = row['file_path']
                stem = (os.path.basename(fp)
                        .replace('.tsv.gz', '').replace('.tsv', ''))
                if stem not in drivers_by_file:
                    continue

                gt = drivers_by_file[stem]
                ranked = model.score_sequences(fp)

                for k in ks:
                    top_k = {cdr3 for cdr3, _ in ranked[:k]}
                    hits = top_k & gt
                    precision = len(hits) / k
                    recall = len(hits) / len(gt) if gt else 0.0

                    fold_precisions[k].append(precision)
                    fold_recalls[k].append(recall)
                    all_results.append({
                        'fold': test_fold,
                        'k': k,
                        'specimen_label': row['specimen_label'],
                        'participant_label': row[participant_col],
                        'disease_label': int(row['label']),
                        'filename': stem,
                        'n_repertoire_unique_cdr3s': len(ranked),
                        'n_ground_truth_drivers': len(gt),
                        'n_hits_at_k': len(hits),
                        'precision_at_k': precision,
                        'recall_at_k': recall,
                    })

            n_reps = len(fold_precisions[ks[0]])
            if n_reps > 0:
                print(f"\nFold {test_fold}: {n_reps} repertoires")
                for k in ks:
                    mp = np.mean(fold_precisions[k])
                    mr = np.mean(fold_recalls[k])
                    print(f"  Mean Precision@{k}: {mp:.4f}")
                    print(f"  Mean Recall@{k}:    {mr:.4f}")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': n_reps,
                    **{f'mean_precision_at_{k}': np.mean(fold_precisions[k]) for k in ks},
                    **{f'mean_recall_at_{k}': np.mean(fold_recalls[k]) for k in ks},
                })
            else:
                print(f"  Fold {test_fold}: no test repertoires with ground truth drivers")
                fold_summaries.append({
                    'fold': test_fold, 'n_repertoires': 0,
                    **{f'mean_precision_at_{k}': float('nan') for k in ks},
                    **{f'mean_recall_at_{k}': float('nan') for k in ks},
                })

        results_df = pd.DataFrame(all_results)

        print(f"\n{'=' * 60}")
        print(f"OVERALL RESULTS: {target_disease} Driver Identification")
        print(f"{'=' * 60}")

        for k in ks:
            k_df = results_df[results_df['k'] == k] if len(results_df) > 0 else results_df
            if len(k_df) > 0:
                overall_prec = k_df['precision_at_k'].mean()
                overall_rec  = k_df['recall_at_k'].mean()
                total_hits   = k_df['n_hits_at_k'].sum()
                total_gt     = k_df['n_ground_truth_drivers'].sum()
                print(f"\nk={k} ({len(k_df)} repertoires):")
                print(f"  Overall Precision@{k} (macro): {overall_prec:.4f}")
                print(f"  Overall Recall@{k}    (macro): {overall_rec:.4f}")
                if total_gt > 0:
                    print(f"  Overall Precision@{k} (micro): "
                          f"{total_hits / (len(k_df) * k):.4f}")
                    print(f"  Overall Recall@{k}    (micro): "
                          f"{total_hits / total_gt:.4f}")
            else:
                print(f"\nk={k}: No results")

        print(f"\nPer-fold breakdown:")
        for s in fold_summaries:
            k_strs = '  '.join(
                f"P@{k}={s[f'mean_precision_at_{k}']:.4f} R@{k}={s[f'mean_recall_at_{k}']:.4f}"
                for k in ks
            )
            print(f"  Fold {s['fold']}: {s['n_repertoires']} reps | {k_strs}")

        if output_csv and len(results_df) > 0:
            results_df.to_csv(output_csv, index=False)
            print(f"\nPer-repertoire results saved to: {output_csv}")

        return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ensemble XGBoost Driver Sequence Identification'
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True,
                        help='Ground truth driver sequences CSV '
                             '(columns: disease, filename, sample_cdr3)')
    parser.add_argument('--k', type=str, required=True,
                        help='Comma-separated k values, e.g. 100,1000,10000')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='Directory written by --model_save_dir in '
                             'ensemble_xgboost_disease_classification.py')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--n_cv_folds', type=int, default=3)
    parser.add_argument('--max_repertoires', type=int, default=None,
                        help='Subsample to this many repertoires total (for debug runs).')
    parser.add_argument('--ext_metadata_path', type=str, default=None,
                        help='Optional external-cohort metadata TSV (MAL-ID column style).')
    parser.add_argument('--ext_data_dir', type=str, default=None,
                        help='Directory of external repertoire files.')
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv',
                        help='Filename template for external repertoires.')
    args = parser.parse_args()
    ks = [int(x) for x in args.k.split(',')]

    evaluator = EnsembleXGBoostDriverEvaluator(
        model_save_dir=args.model_save_dir,
    )

    evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        ks=ks,
        n_folds=args.n_cv_folds,
        max_repertoires=args.max_repertoires,
        output_csv=args.output_csv,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_file_template=args.ext_file_template,
    )
