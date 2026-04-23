"""
Evaluation script for the MotifBoost disease classification model.

Provides cross-validation functionality for evaluating MotifBoostClassifier
on binary disease vs. Healthy/Background classification tasks.

MotifBoost extracts gapped n-gram (default: 3-gram + 4-gram) features from
CDR3 amino acid sequences weighted by clone counts, then trains an
Optuna-tuned LightGBM classifier.

Reference:
    Koike-Akino et al. 2021, "MotifBoost: A Machine Learning Method for
    Adaptive Immune Repertoire Classification"
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, f1_score)
from tqdm import tqdm

from models.MotifBoost.motifboost import MotifBoost
from utils.covariate_residualization import covariate_adjusted_predict, filter_complete_demographics
from utils.cohort_adjustments import apply_cohort_adjustment


class MotifBoostEvaluator:
    """
    Evaluator for MotifBoost on binary disease classification.

    The cross-validation loop follows the pre-defined fold assignments in
    the metadata (malid_cross_validation_fold_id_when_in_test_set).
    Hyperparameter search (Optuna) is handled internally by the model.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(
        self,
        sequence_col='cdr3_aa',
        ngram_range=(3, 4),
        classifier_method='optuna-lightgbm',
        count_weight_mode=True,
        tfidf_mode=False,
        augmentation_times=5,
        augmentation_rate=0.5,
        n_jobs=None,
        subsample_fraction=1.0,
        subsample_seed=7,
    ):
        self.sequence_col = sequence_col
        self.ngram_range = ngram_range
        self.classifier_method = classifier_method
        self.count_weight_mode = count_weight_mode
        self.tfidf_mode = tfidf_mode
        self.augmentation_times = augmentation_times
        self.augmentation_rate = augmentation_rate
        self.n_jobs = n_jobs
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed

    # ------------------------------------------------------------------
    # Metadata helpers (mirrors EnsembleRegressionEvaluator)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             adjust_distribution_by_demographics=False,
                             random_baseline=False,
                             random_baseline_seed=7):
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
        return MotifBoost(
            sequence_col=self.sequence_col,
            ngram_range=self.ngram_range,
            classifier_method=self.classifier_method,
            count_weight_mode=self.count_weight_mode,
            tfidf_mode=self.tfidf_mode,
            augmentation_times=self.augmentation_times,
            augmentation_rate=self.augmentation_rate,
            n_jobs=self.n_jobs,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(
        self,
        metadata_path,
        target_disease,
        data_dir,
        participant_col='participant_label',
        file_prefix='part_table_',
        file_suffix='.tsv.gz',
        disease_col='disease',
        fold_col='malid_cross_validation_fold_id_when_in_test_set',
        n_folds=3,
        allowed_participants=None,
        adjust_distribution_by_demographics=False,
        random_baseline=False,
        random_baseline_seed=7,
        covariate_adjust=False,
        debug_repertoires=0,
        output_csv=None,
    ):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold, all non-test samples are passed to MotifBoost.train().
        Optuna hyperparameter search runs inside that call.

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease name to classify against Healthy/Background.
            data_dir: Directory containing repertoire .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix / file_suffix: Filename construction parameters.
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined fold IDs (0, 1, 2).
            n_folds: Number of CV folds (default: 3).
            allowed_participants: Optional set of specimen labels to restrict to.
            adjust_distribution_by_demographics: If True, apply per-disease
                cohort balancing (age-matching or ancestry filter).
            covariate_adjust: If True, residualize model scores against
                demographics and fit an L1 logistic regression head.
            debug_repertoires: If > 0, subsample to N disease + N healthy
                repertoires for fast debug runs.
            output_csv: Optional path to write per-sample scores CSV.

        Returns:
            pd.DataFrame with per-sample scores across all folds.
        """
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
            print(f"  Trained on {train_result['n_train']} repertoires.")

            if covariate_adjust:
                # ----------------------------------------------------------
                # Score-level covariate adjustment
                # ----------------------------------------------------------
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
                method_name = 'MotifBoost_CovAdj'
                if random_baseline:
                    method_name += '_RandomBaseline'

                for (_, row), score in zip(test_cov.iterrows(), test_probs):
                    entry = {
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': method_name,
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    }
                    if random_baseline:
                        entry['random_baseline_seed'] = int(random_baseline_seed)
                    all_test_rows.append(entry)
            else:
                # ----------------------------------------------------------
                # Standard prediction
                # ----------------------------------------------------------
                test_probs = np.array([
                    model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(test_data['file_path'].tolist(), desc="Testing")
                ])
                test_labels_arr = test_data['label'].values

                method_name = 'MotifBoost'
                if random_baseline:
                    method_name += '_RandomBaseline'
                for (_, row), score in zip(test_data.iterrows(), test_probs):
                    entry = {
                        'participant_label': row[participant_col],
                        'specimen_label': row['specimen_label'],
                        'disease_label': int(row['label']),
                        'disease_label_str': row[disease_col],
                        'method': method_name,
                        'disease_model': target_disease,
                        'model_score': float(score),
                        'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    }
                    if random_baseline:
                        entry['random_baseline_seed'] = int(random_baseline_seed)
                    all_test_rows.append(entry)

            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
            test_f1 = f1_score(test_labels_arr, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            fold_results.append({
                'fold': test_fold,
                'n_train': train_result['n_train'],
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
        description='MotifBoost Disease Classification Evaluation'
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--ngram_range', type=int, nargs=2, default=[3, 4],
                        metavar=('MIN', 'MAX'),
                        help='N-gram range (default: 3 4)')
    parser.add_argument('--classifier_method', type=str,
                        default='optuna-lightgbm',
                        choices=['optuna-lightgbm', 'lightgbm',
                                 'linear_regression', 'svm'],
                        help='Classifier backend (default: optuna-lightgbm)')
    parser.add_argument('--augmentation_times', type=int, default=5,
                        help='Data augmentation multiplier (default: 5)')
    parser.add_argument('--augmentation_rate', type=float, default=0.5,
                        help='Subsample rate per augmentation (default: 0.5)')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Parallel jobs for feature extraction '
                             '(default: auto)')
    parser.add_argument('--adjust_distribution_by_demographics',
                        action='store_true',
                        help='Apply per-disease cohort balancing')
    parser.add_argument('--random_baseline_seeds', type=int, nargs='+', default=None,
                        help='Run the random-sampling healthy baseline for each seed '
                             '(implies --adjust_distribution_by_demographics). For each '
                             'seed, healthy is resampled uniformly at random to the same '
                             'target N as the demographic-matched cohort; disease side '
                             'mirrors the demographic-matched run. Results from all seeds '
                             'are concatenated into one output, with a '
                             '`random_baseline_seed` column. Example: 7 14 21 28 35.')
    parser.add_argument('--covariate_adjust', action='store_true',
                        help='Residualize scores against demographics '
                             'and fit L1 logistic regression head')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--debug_repertoires', type=int, default=0,
                        help='Subsample to N disease + N healthy for fast '
                             'debug runs (0 = disabled)')
    args = parser.parse_args()

    evaluator = MotifBoostEvaluator(
        ngram_range=tuple(args.ngram_range),
        classifier_method=args.classifier_method,
        augmentation_times=args.augmentation_times,
        augmentation_rate=args.augmentation_rate,
        n_jobs=args.n_jobs,
    )

    if args.random_baseline_seeds:
        seed_dfs = []
        for seed in args.random_baseline_seeds:
            print(f"\n{'#' * 60}")
            print(f"# RANDOM BASELINE RUN — seed={seed}")
            print(f"{'#' * 60}")
            seed_df = evaluator.run_cross_validation(
                metadata_path=args.metadata_path,
                target_disease=args.target_disease,
                data_dir=args.repertoire_data_dir,
                adjust_distribution_by_demographics=True,
                random_baseline=True,
                random_baseline_seed=seed,
                covariate_adjust=args.covariate_adjust,
                debug_repertoires=args.debug_repertoires,
                output_csv=None,
            )
            seed_dfs.append(seed_df)
        scores_df = pd.concat(seed_dfs, axis=0, ignore_index=True)

        # Aggregate across seeds: one (AUROC, AUPR) per seed, then mean ± std.
        per_seed = []
        for seed, seed_df in scores_df.groupby('random_baseline_seed'):
            y = seed_df['disease_label'].values
            p = seed_df['model_score'].values
            per_seed.append({
                'random_baseline_seed': int(seed),
                'overall_auroc': roc_auc_score(y, p),
                'overall_aupr': average_precision_score(y, p),
            })
        summary_df = pd.DataFrame(per_seed)
        print(f"\n{'#' * 60}")
        print(f"# RANDOM BASELINE SUMMARY — across {len(summary_df)} seeds")
        print(f"{'#' * 60}")
        print(summary_df.to_string(index=False))
        print(f"Mean overall AUROC: {summary_df['overall_auroc'].mean():.4f} "
              f"± {summary_df['overall_auroc'].std(ddof=0):.4f}")
        print(f"Mean overall AUPR:  {summary_df['overall_aupr'].mean():.4f} "
              f"± {summary_df['overall_aupr'].std(ddof=0):.4f}")

        if args.output_csv and len(scores_df) > 0:
            scores_df.to_csv(args.output_csv, index=False)
            print(f"\nAll-seed scores saved to: {args.output_csv}")
    else:
        scores_df = evaluator.run_cross_validation(
            metadata_path=args.metadata_path,
            target_disease=args.target_disease,
            data_dir=args.repertoire_data_dir,
            adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
            covariate_adjust=args.covariate_adjust,
            debug_repertoires=args.debug_repertoires,
            output_csv=args.output_csv,
        )
