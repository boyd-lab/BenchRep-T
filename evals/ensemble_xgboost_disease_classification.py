"""
Evaluation script for the Ensemble XGBoost (Gapped k-mer + V/J gene) disease classification model.

Same cross-validation structure and output format as ensemble_regression_disease_classification.py.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, f1_score)
from tqdm import tqdm

from models.ensemble_xgboost import XGBoostKmer
from utils.covariate_residualization import covariate_adjusted_predict, filter_complete_demographics
from utils.cohort_adjustments import apply_cohort_adjustment


SUBMODEL_SUFFIXES = {
    'ensemble': '',
    'kmer_only': '_Kmer',
    'vj_only': '_VJ',
}


class EnsembleXGBoostEvaluator:

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, n_cv_folds=3, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7,
                 kmer_size=4, use_gaps=True, submodel='ensemble', n_jobs=None):
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.kmer_size = kmer_size
        self.use_gaps = use_gaps
        self.submodel = submodel
        self.n_jobs = n_jobs

    def _method_name(self):
        name = f'EnsembleXGBoost_{self.kmer_size}mer'
        if self.use_gaps:
            name += '_gapped'
        name += SUBMODEL_SUFFIXES[self.submodel]
        return name

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             adjust_distribution_by_demographics=False,
                             random_baseline=False, random_baseline_seed=7):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        if adjust_distribution_by_demographics:
            filtered = apply_cohort_adjustment(
                filtered, target_disease,
                seed=random_baseline_seed if random_baseline else self.subsample_seed,
                random_baseline=random_baseline,
            )

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(filtered)}")
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
            print(f"Note: {missing} of {original} files not found; "
                  f"proceeding with {len(filtered)}.")
        return filtered

    def _make_model(self):
        return XGBoostKmer(
            val_split=self.val_split,
            n_cv_folds=self.n_cv_folds,
            sequence_col=self.sequence_col,
            v_gene_col=self.v_gene_col,
            j_gene_col=self.j_gene_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            kmer_size=self.kmer_size,
            use_gaps=self.use_gaps,
            submodel=self.submodel,
            n_jobs=self.n_jobs,
        )

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                             participant_col='participant_label',
                             file_prefix='part_table_', file_suffix='.tsv.gz',
                             disease_col='disease',
                             fold_col='malid_cross_validation_fold_id_when_in_test_set',
                             n_folds=3, allowed_participants=None,
                             adjust_distribution_by_demographics=False,
                             random_baseline=False, random_baseline_seed=7,
                             covariate_adjust=False,
                             debug_repertoires=0, output_csv=None):
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(
            raw_metadata, target_disease, disease_col,
            adjust_distribution_by_demographics=adjust_distribution_by_demographics,
            random_baseline=random_baseline,
            random_baseline_seed=random_baseline_seed,
        )
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if debug_repertoires > 0:
            disease_rows = metadata[metadata['label'] == 1].head(debug_repertoires)
            healthy_rows = metadata[metadata['label'] == 0].head(debug_repertoires)
            metadata = pd.concat([disease_rows, healthy_rows])
            print(f"Debug: subsampled to {len(disease_rows)} disease + "
                  f"{len(healthy_rows)} healthy repertoires")

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants set.")

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_data = metadata[~test_mask]
            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            model = self._make_model()
            train_result = model.train(
                train_data['file_path'].tolist(),
                train_data['label'].tolist(),
            )
            print(f"  alpha={train_result['best_alpha']:.1f}, "
                  f"val AUROC={train_result['val_auroc']:.4f}")

            if covariate_adjust:
                tv_cov = filter_complete_demographics(train_data)
                test_cov = filter_complete_demographics(test_data)
                print(f"  Covariate adjust: {len(tv_cov)} train, "
                      f"{len(test_cov)} test samples with usable demographics.")

                tv_scores = [
                    model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(tv_cov['file_path'].tolist(), desc="Scoring train")
                ]
                test_scores = [
                    model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(test_cov['file_path'].tolist(), desc="Scoring test")
                ]
                X_tv = np.array(tv_scores).reshape(-1, 1)
                X_test_emb = np.array(test_scores).reshape(-1, 1)
                test_probs = covariate_adjusted_predict(
                    X_tv, tv_cov, tv_cov['label'].values, X_test_emb, test_cov
                )
                test_labels_arr = test_cov['label'].values
                method_name = self._method_name() + '_CovAdj'

                for (_, row), score in zip(test_cov.iterrows(), test_probs):
                    all_test_rows.append({
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': method_name,
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    })
            else:
                test_probs = np.array([
                    model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(test_data['file_path'].tolist(), desc="Testing")
                ])
                test_labels_arr = test_data['label'].values

                for (_, row), score in zip(test_data.iterrows(), test_probs):
                    all_test_rows.append({
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': self._method_name(),
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    })

            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
            test_f1 = f1_score(test_labels_arr, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            fold_results.append({
                'fold': test_fold,
                'best_alpha': train_result['best_alpha'],
                'val_auroc': train_result['val_auroc'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels_arr.tolist())

            del model

        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        print(f"\n{'=' * 60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'=' * 60}")
        fold_aurocs = [r['test_auroc'] for r in fold_results]
        fold_auprs = [r['test_aupr'] for r in fold_results]
        fold_balanced_accs = [r['test_balanced_acc'] for r in fold_results]
        fold_f1s = [r['test_f1'] for r in fold_results]
        print(f"Mean Test AUROC:        {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:         {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
        print(f"Mean Test Balanced Acc: {np.mean(fold_balanced_accs):.4f} ± {np.std(fold_balanced_accs):.4f}")
        print(f"Mean Test F1:           {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"Overall AUROC (all folds combined):        {overall_auroc:.4f}")
        print(f"Overall AUPR  (all folds combined):        {overall_aupr:.4f}")
        print(f"Overall Balanced Acc (all folds combined): {overall_balanced_acc:.4f}")
        print(f"Overall F1 (all folds combined):           {overall_f1:.4f}")

        scores_df = pd.DataFrame(all_test_rows)
        if output_csv and len(scores_df) > 0:
            scores_df.to_csv(output_csv, index=False)
            print(f"\nScores saved to: {output_csv}")

        return scores_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ensemble XGBoost (Gapped k-mer + V/J gene) Disease Classification'
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--kmer_size', type=int, default=4)
    parser.add_argument('--no_gaps', action='store_true')
    parser.add_argument('--submodel', type=str, default='ensemble',
                        choices=['ensemble', 'kmer_only', 'vj_only'])
    parser.add_argument('--n_cv_folds', type=int, default=3)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--adjust_distribution_by_demographics', action='store_true')
    parser.add_argument('--covariate_adjust', action='store_true')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--debug_repertoires', type=int, default=0)
    args = parser.parse_args()

    evaluator = EnsembleXGBoostEvaluator(
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        kmer_size=args.kmer_size,
        use_gaps=not args.no_gaps,
        submodel=args.submodel,
        n_jobs=args.n_jobs,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
        covariate_adjust=args.covariate_adjust,
        debug_repertoires=args.debug_repertoires,
        output_csv=args.output_csv,
    )
