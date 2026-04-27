"""
V/J gene + demographics disease classification.

Single-stage classifier that concatenates per-sample V/J gene count features
(as used by the V/J sub-model of `Gapped_4mer_VJgene`) with demographic
features (age, sex, ancestry), then fits one L1-regularized logistic
regression to predict disease status.

Tests whether V/J gene usage and demographics carry jointly predictive
signal, without the two-stage stacking architecture.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.ensemble_regression import Gapped_4mer_VJgene


class VJDemographicsEvaluator:
    """
    Concatenate V/J gene counts with demographic features (age, sex, ancestry)
    and train a single L1 logistic regression per CV fold.
    """

    HEALTHY_LABEL = "Healthy/Background"
    C_CANDIDATES = [1.0, 0.2, 0.1, 0.05, 0.03]

    def __init__(self, n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, indices_map=None):
        """
        Args:
            n_cv_folds: CV folds used for C tuning.
            sequence_col/v_gene_col/j_gene_col: Column names for repertoire data.
            subsample_fraction/subsample_seed/indices_map: For depth experiments.
        """
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.indices_map = indices_map
        self.canonicalize_genes = False

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
        """Convert demographics into a numeric feature matrix."""
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

    def _build_vj_feature_extractor(self):
        """Reuse Gapped_4mer_VJgene only for its V/J feature dict extraction."""
        return Gapped_4mer_VJgene(
            sequence_col=self.sequence_col,
            v_gene_col=self.v_gene_col,
            j_gene_col=self.j_gene_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            indices_map=self.indices_map,
            canonicalize_genes=self.canonicalize_genes,
            submodel='vj_only',
        )

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
                clf = LogisticRegression(C=c, penalty='l1', solver='liblinear',
                                         max_iter=1000)
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
                              n_folds=3,
                              ext_metadata_path=None, ext_data_dir=None,
                              ext_file_template='{participant_label}_TCRB.tsv'):
        """
        Run 3-fold CV. Per fold: fit DictVectorizer + StandardScaler on the
        non-test V/J counts, hstack with demographic features, train one L1
        logistic regression, evaluate on the held-out test fold.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease,
                                             disease_col)
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
            self.canonicalize_genes = True

        all_test_rows = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"V/J + DEMO FOLD {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_data = metadata[~test_mask]

            print(f"Train: {len(train_data)}    Test: {len(test_data)}")

            # --- V/J gene count features ---
            vj_extractor = self._build_vj_feature_extractor()
            print("\nExtracting V/J counts (train)...")
            train_vj_dicts = [vj_extractor._get_vj_feature_dict(fp)
                              for fp in tqdm(train_data['file_path'].tolist(),
                                             leave=False)]
            print("Extracting V/J counts (test)...")
            test_vj_dicts = [vj_extractor._get_vj_feature_dict(fp)
                             for fp in tqdm(test_data['file_path'].tolist(),
                                            leave=False)]

            vectorizer = DictVectorizer(sparse=True)
            X_train_vj = vectorizer.fit_transform(train_vj_dicts)
            X_test_vj = vectorizer.transform(test_vj_dicts)

            vj_scaler = StandardScaler(with_mean=False)
            X_train_vj = vj_scaler.fit_transform(X_train_vj)
            X_test_vj = vj_scaler.transform(X_test_vj)

            vj_feature_names = list(vectorizer.get_feature_names_out())
            n_vj = len(vj_feature_names)

            # --- Demographic features (ancestry cats from training data) ---
            ancestry_cats = sorted(train_data['ancestry'].unique().tolist())
            X_train_demo, _, demo_feature_names = self.featurize_demographics(
                train_data, ancestry_categories=ancestry_cats)
            X_test_demo, _, _ = self.featurize_demographics(
                test_data, ancestry_categories=ancestry_cats)

            demo_scaler = StandardScaler()
            X_train_demo = demo_scaler.fit_transform(X_train_demo)
            X_test_demo = demo_scaler.transform(X_test_demo)

            # --- Concatenate V/J + demographics ---
            X_train = sparse_hstack([X_train_vj, csr_matrix(X_train_demo)],
                                    format='csr')
            X_test = sparse_hstack([X_test_vj, csr_matrix(X_test_demo)],
                                   format='csr')

            y_train = train_data['label'].values
            y_test = test_data['label'].values

            all_feature_names = vj_feature_names + demo_feature_names
            print(f"\nFeatures: {n_vj} V/J + {len(demo_feature_names)} demo "
                  f"= {len(all_feature_names)} total")

            # --- Tune C and fit ---
            print("\nTuning C via stratified CV...")
            best_c = self._tune_c_cv(X_train, y_train, self.C_CANDIDATES,
                                     n_folds=self.n_cv_folds)
            print(f"  Best C: {best_c}")

            model = LogisticRegression(C=best_c, penalty='l1',
                                       solver='liblinear', max_iter=1000)
            model.fit(X_train, y_train)

            n_nonzero = int(np.sum(model.coef_.ravel() != 0))
            print(f"  Non-zero coefficients: {n_nonzero} / {len(all_feature_names)}")

            # --- Evaluate ---
            test_probs = model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, test_probs)
            aupr = average_precision_score(y_test, test_probs)
            preds = (test_probs >= 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"\n--- Fold {test_fold} Test Results ---")
            print(f"  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                  f"Balanced Acc={balanced_acc:.4f}  F1={f1:.4f}")

            # Log demographic coefficients (always interesting; V/J coefs
            # are sparse and there are many, so summarize counts only)
            coefs = model.coef_.ravel()
            demo_coefs = dict(zip(demo_feature_names, coefs[n_vj:]))
            vj_nonzero = int(np.sum(coefs[:n_vj] != 0))
            print(f"\n  Non-zero V/J features: {vj_nonzero} / {n_vj}")
            print(f"  Demographic coefficients:")
            for name, coef in demo_coefs.items():
                print(f"    {name}: {coef:+.4f}")

            for (_, row), s in zip(test_data.iterrows(), test_probs):
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'disease_model': target_disease,
                    'malid_cross_validation_fold_id_when_in_test_set':
                        test_fold,
                    'method': 'VJ_Demo',
                    'model_score': float(s),
                })

            fold_results.append({
                'fold': test_fold,
                'auroc': auroc, 'aupr': aupr,
                'balanced_acc': balanced_acc, 'f1': f1,
                'best_c': best_c,
                'n_vj_features': n_vj,
                'n_vj_nonzero': vj_nonzero,
                'demo_coefficients': demo_coefs,
            })

            vj_extractor.clear_cache()

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
        overall_preds = (scores_df['model_score'].values >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(
            scores_df['disease_label'].values, overall_preds)
        overall_f1 = f1_score(scores_df['disease_label'].values, overall_preds)
        print(f"  V/J + Demo  AUROC={overall_auroc:.4f}  AUPR={overall_aupr:.4f}  "
              f"Balanced Acc={overall_balanced_acc:.4f}  F1={overall_f1:.4f}")

        print(f"\nPer-fold results:")
        print(f"  {'Fold':<6} {'AUROC':<12} {'AUPR':<12} {'Bal Acc':<12} {'F1':<12}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['auroc']:<12.4f} {r['aupr']:<12.4f} "
                  f"{r['balanced_acc']:<12.4f} {r['f1']:<12.4f}")

        aurocs = [r['auroc'] for r in fold_results]
        auprs = [r['aupr'] for r in fold_results]
        balanced_accs = [r['balanced_acc'] for r in fold_results]
        f1s = [r['f1'] for r in fold_results]
        print(f"\nMean ± Std:")
        print(f"  AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}  "
              f"AUPR={np.mean(auprs):.4f}±{np.std(auprs):.4f}  "
              f"Balanced Acc={np.mean(balanced_accs):.4f}±{np.std(balanced_accs):.4f}  "
              f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

        return scores_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V/J gene counts + Demographics disease classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--ext_metadata_path', type=str, default=None,
                        help='Optional external-cohort metadata TSV (MAL-ID column style).')
    parser.add_argument('--ext_data_dir', type=str, default=None,
                        help='Directory of external repertoire files.')
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv',
                        help='Filename template for external repertoires.')
    args = parser.parse_args()

    evaluator = VJDemographicsEvaluator()

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_file_template=args.ext_file_template,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
