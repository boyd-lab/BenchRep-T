"""
Meta-model evaluator combining ensemble regression predictions with demographic features.

Tests whether TCR repertoire features (gapped 4-mer + V/J gene model predictions)
and demographic features (age, sex, ancestry) are orthogonal for disease
classification by training a stacking meta-model.

Architecture per CV fold:
  1. Split non-test data into base_train (80%) and meta_train (20%)
  2. Train ensemble regression on base_train
  3. Obtain out-of-sample ensemble regression predictions on meta_train and test
  4. Extract demographic features for meta_train and test
  5. Train logistic regression meta-model on combined features
  6. Evaluate meta-model on test set
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from models.ensemble_regression import Gapped_4mer_VJgene


class MetaModelEvaluator:
    """
    Stacking meta-model: ML baseline (gapped 4-mer + V/J gene) predictions
    combined with demographic features (age, sex, ancestry).

    Meta-features per sample: [ml_prob, kmer_prob, vj_prob, age, sex, ancestry...]

    All samples are required to have complete demographic data (age, sex,
    ancestry), so the comparison between methods is on the same subset.
    """

    HEALTHY_LABEL = "Healthy/Background"
    META_C_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    def __init__(self, base_train_fraction=0.8, ml_val_split=0.2,
                 ml_n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, indices_map=None,
                 submodel='ensemble'):
        """
        Args:
            base_train_fraction: Fraction of non-test data used to train the
                ML baseline. Remaining fraction trains the meta-model.
            ml_val_split: Internal val fraction used by ML baseline for alpha tuning.
            ml_n_cv_folds: CV folds used by ML baseline for C tuning.
            sequence_col/v_gene_col/j_gene_col: Column names for repertoire data.
            subsample_fraction/subsample_seed/indices_map: For depth experiments.
            submodel: 'ensemble' (default), 'kmer_only', or 'vj_only'.
        """
        self.base_train_fraction = base_train_fraction
        self.ml_val_split = ml_val_split
        self.ml_n_cv_folds = ml_n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.indices_map = indices_map
        self.submodel = submodel

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        """Filter to target disease vs healthy, requiring complete demographics."""
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        before = len(filtered)
        filtered = filtered.dropna(subset=['age', 'sex', 'ancestry'])
        filtered = filtered[filtered['ancestry'].str.strip() != '']
        after = len(filtered)

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {after} (dropped {before - after} with missing demographics)")
        return filtered

    def construct_file_path(self, participant_label, specimen_label, data_dir,
                            file_prefix='part_table_', file_suffix='.tsv.gz'):
        return os.path.join(
            data_dir,
            f"{file_prefix}{participant_label}_{specimen_label}{file_suffix}")

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: self.construct_file_path(
                row[participant_col], row['specimen_label'],
                data_dir, file_prefix, file_suffix
            ), axis=1
        )
        return metadata

    def filter_existing_files(self, metadata):
        original_count = len(metadata)
        metadata = metadata.copy()
        metadata['file_exists'] = metadata['file_path'].apply(os.path.exists)
        filtered = metadata[metadata['file_exists']].drop(columns=['file_exists'])
        missing = original_count - len(filtered)
        if missing > 0:
            print(f"Note: {missing} of {original_count} files not found; "
                  f"proceeding with {len(filtered)}.")
        return filtered

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def featurize_demographics(self, data, ancestry_categories=None):
        """
        Convert demographic columns into a numeric feature matrix.

        Returns:
            (feature_matrix, ancestry_categories, feature_names)
        """
        age = data['age'].values.astype(float)
        sex = (data['sex'] == 'M').astype(int).values

        if ancestry_categories is None:
            ancestry_categories = sorted(data['ancestry'].unique().tolist())

        ancestry_dummies = np.zeros((len(data), len(ancestry_categories)),
                                    dtype=float)
        for i, cat in enumerate(ancestry_categories):
            ancestry_dummies[:, i] = (data['ancestry'].values == cat).astype(float)

        features = np.column_stack([age, sex, ancestry_dummies])
        feature_names = (['age', 'sex']
                         + [f'ancestry_{cat}' for cat in ancestry_categories])
        return features, ancestry_categories, feature_names

    def _get_ml_predictions(self, ml_model, file_paths):
        """Get ML baseline predictions for each file.

        Returns an (N, k) array where k depends on the submodel:
          - ensemble: [ml_prob, kmer_prob, vj_prob]
          - kmer_only: [kmer_prob]
          - vj_only: [vj_prob]
        """
        preds = []
        for fp in tqdm(file_paths, desc="ML predictions", leave=False):
            result = ml_model.predict_diagnosis(fp)
            if self.submodel == 'ensemble':
                preds.append([
                    result['probability_positive'],
                    result['kmer_probability'],
                    result['vj_probability'],
                ])
            else:
                preds.append([result['probability_positive']])
        return np.array(preds)

    def _tune_c_cv(self, X, y, c_candidates, n_folds=5):
        """Tune C via stratified CV. Falls back to C=1.0 if too few samples."""
        n_per_class = min(np.sum(y == 0), np.sum(y == 1))
        actual_folds = min(n_folds, n_per_class)
        if actual_folds < 2:
            print("  Warning: too few samples per class for CV; using C=1.0")
            return 1.0

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True,
                              random_state=self.subsample_seed)
        best_c, best_auroc = c_candidates[0], -1.0

        for c in c_candidates:
            fold_aurocs = []
            for tr_idx, vl_idx in skf.split(X, y):
                if len(np.unique(y[vl_idx])) < 2:
                    continue
                clf = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
                clf.fit(X[tr_idx], y[tr_idx])
                probs = clf.predict_proba(X[vl_idx])[:, 1]
                fold_aurocs.append(roc_auc_score(y[vl_idx], probs))

            if fold_aurocs:
                mean_auroc = np.mean(fold_aurocs)
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_c = c

        return best_c

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7):
        """
        Run 3-fold CV for the stacking meta-model.

        For each fold, non-test data is split into base_train
        (``base_train_fraction``, default 80 %) used to train the ML baseline,
        and meta_train (remaining 20 %) used to train the meta logistic
        regression on combined features.  The meta-model is evaluated on
        the held-out test fold.

        Returns:
            DataFrame with per-sample test predictions for ``Meta_ML_Demo``.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease,
                                             disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                        file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        all_test_rows = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"META-MODEL FOLD {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            nontest_data = metadata[~test_mask]

            # Split non-test into base_train (ML baseline) and meta_train
            base_train, meta_train = train_test_split(
                nontest_data,
                train_size=self.base_train_fraction,
                random_state=random_state,
                stratify=nontest_data['label']
            )

            print(f"Base train (ML baseline): {len(base_train)}")
            print(f"Meta train (meta-model):  {len(meta_train)}")
            print(f"Test:                     {len(test_data)}")

            # --- Train ML baseline on base_train ---
            print("\n--- Training ML Baseline on base_train ---")
            ml_model = Gapped_4mer_VJgene(
                val_split=self.ml_val_split,
                n_cv_folds=self.ml_n_cv_folds,
                sequence_col=self.sequence_col,
                v_gene_col=self.v_gene_col,
                j_gene_col=self.j_gene_col,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed,
                indices_map=self.indices_map,
                submodel=self.submodel,
            )
            ml_train_result = ml_model.train(
                base_train['file_path'].tolist(),
                base_train['label'].tolist()
            )

            # --- ML predictions on meta_train and test (out-of-sample) ---
            print("\nML baseline predictions on meta_train...")
            meta_ml_preds = self._get_ml_predictions(
                ml_model, meta_train['file_path'].tolist())

            print("ML baseline predictions on test...")
            test_ml_preds = self._get_ml_predictions(
                ml_model, test_data['file_path'].tolist())

            # --- Demographic features ---
            # Derive ancestry categories from all non-test data for consistency
            ancestry_cats = sorted(nontest_data['ancestry'].unique().tolist())

            X_meta_demo, _, demo_feature_names = self.featurize_demographics(
                meta_train, ancestry_categories=ancestry_cats)
            X_test_demo, _, _ = self.featurize_demographics(
                test_data, ancestry_categories=ancestry_cats)

            y_meta = meta_train['label'].values
            y_test = test_data['label'].values

            # --- Build meta-features ---
            if self.submodel == 'ensemble':
                ml_feature_names = ['ml_prob', 'kmer_prob', 'vj_prob']
            elif self.submodel == 'kmer_only':
                ml_feature_names = ['kmer_prob']
            else:
                ml_feature_names = ['vj_prob']
            all_feature_names = ml_feature_names + demo_feature_names

            X_meta_combined = np.hstack([meta_ml_preds, X_meta_demo])
            X_test_combined = np.hstack([test_ml_preds, X_test_demo])

            print(f"\nMeta-features ({len(all_feature_names)}): "
                  f"{all_feature_names}")

            # --- Train meta-model (combined) ---
            print("\n--- Training Meta-Model (ML + Demographics) ---")
            best_meta_c = self._tune_c_cv(
                X_meta_combined, y_meta, self.META_C_CANDIDATES)
            print(f"  Best C (meta): {best_meta_c}")

            meta_model = LogisticRegression(
                C=best_meta_c, max_iter=1000, solver='lbfgs')
            meta_model.fit(X_meta_combined, y_meta)

            # --- Evaluate meta-model on test ---
            test_meta_probs = meta_model.predict_proba(X_test_combined)[:, 1]

            meta_auroc = roc_auc_score(y_test, test_meta_probs)
            meta_aupr = average_precision_score(y_test, test_meta_probs)

            print(f"\n--- Fold {test_fold} Test Results ---")
            print(f"  Meta (combined): AUROC={meta_auroc:.4f}  AUPR={meta_aupr:.4f}")

            # Log meta-model coefficients
            meta_coefs = dict(zip(all_feature_names, meta_model.coef_[0]))
            print(f"\n  Meta-model coefficients:")
            for name, coef in meta_coefs.items():
                print(f"    {name}: {coef:+.4f}")

            # Store per-sample predictions for meta-model
            for (_, row), meta_s in zip(
                    test_data.iterrows(), test_meta_probs):
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'disease_model': target_disease,
                    'malid_cross_validation_fold_id_when_in_test_set':
                        test_fold,
                    'method': 'Meta_ML_Demo',
                    'model_score': float(meta_s),
                })

            fold_results.append({
                'fold': test_fold,
                'meta_auroc': meta_auroc, 'meta_aupr': meta_aupr,
                'best_meta_c': best_meta_c,
                'meta_coefficients': meta_coefs,
                'ml_train_result': ml_train_result,
            })

            ml_model.clear_cache()

        # ------------------------------------------------------------------
        # Overall summary
        # ------------------------------------------------------------------
        scores_df = pd.DataFrame(all_test_rows)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")

        overall_auroc = roc_auc_score(scores_df['disease_label'],
                                       scores_df['model_score'])
        overall_aupr = average_precision_score(scores_df['disease_label'],
                                                scores_df['model_score'])
        print(f"  Meta (ML + Demo)  AUROC={overall_auroc:.4f}  "
              f"AUPR={overall_aupr:.4f}")

        print(f"\nPer-fold results:")
        print(f"  {'Fold':<6} {'AUROC':<12} {'AUPR':<12}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['meta_auroc']:<12.4f} "
                  f"{r['meta_aupr']:<12.4f}")

        aurocs = [r['meta_auroc'] for r in fold_results]
        auprs = [r['meta_aupr'] for r in fold_results]
        print(f"\nMean ± Std:")
        print(f"  Meta (combined): "
              f"AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
              f"AUPR={np.mean(auprs):.4f}±{np.std(auprs):.4f}")

        return scores_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-Model: ML Baseline + Demographics Disease Classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--base_train_fraction', type=float, default=0.8,
                        help='Fraction of non-test data for ML baseline '
                             '(default: 0.8, remaining 0.2 for meta-model)')
    parser.add_argument('--submodel', type=str, default='ensemble',
                        choices=['ensemble', 'kmer_only', 'vj_only'],
                        help='Ensemble regression sub-model to use as base: '
                             'ensemble (default), kmer_only, or vj_only')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    args = parser.parse_args()

    evaluator = MetaModelEvaluator(
        base_train_fraction=args.base_train_fraction,
        submodel=args.submodel,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
