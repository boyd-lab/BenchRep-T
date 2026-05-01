"""
ABMIL embeddings + demographics disease classification.

Trains ABMIL per cross-validation fold to produce sample-level bag embeddings,
then concatenates them with demographic features (age, sex, ancestry one-hot)
and trains a single L1-regularized logistic regression to predict disease status.

This follows the same pipeline as deeptcr_demographics_disease_classification,
replacing DeepTCR learned embeddings with ABMIL attention-weighted bag embeddings.
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.ensemble_abmil import ABMIL


class ABMILDemographicsEvaluator:
    """
    Concatenate ABMIL sample-level bag embeddings with demographic features
    (age, sex, ancestry) and train a single L1 logistic regression per CV fold.
    """

    HEALTHY_LABEL = "Healthy/Background"
    C_CANDIDATES = [1.0, 0.2, 0.1, 0.05, 0.03]

    def __init__(
        self,
        sequence_col='cdr3_aa',
        v_gene_col='v_call',
        j_gene_col='j_call',
        max_instances=10000,
        M=128,
        L=64,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        patience=10,
        val_split=0.2,
        seed=7,
        subsample_fraction=1.0,
        subsample_seed=7,
        use_gpu=True,
        dropout=0.25,
        max_length=40,
        embedding_dim_aa=64,
        embedding_dim_genes=48,
        kernel=5,
        conv_units=(32, 64, 128),
        features='full',
        n_cv_folds=5,
        indices_map=None,
        debug=False,
        debug_repertoires=10,
    ):
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.max_instances = max_instances
        self.M = M
        self.L = L
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = val_split
        self.seed = seed
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.use_gpu = use_gpu
        self.dropout = dropout
        self.max_length = max_length
        self.embedding_dim_aa = embedding_dim_aa
        self.embedding_dim_genes = embedding_dim_genes
        self.kernel = kernel
        self.conv_units = tuple(conv_units)
        self.features = features
        self.n_cv_folds = n_cv_folds
        self.indices_map = indices_map
        self.debug = debug
        self.debug_repertoires = debug_repertoires
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
        return os.path.join(data_dir,
                            f"{file_prefix}{participant_label}_{specimen_label}{file_suffix}")

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: self.construct_file_path(
                row[participant_col], row['specimen_label'], data_dir,
                file_prefix, file_suffix
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

        ancestry_dummies = np.zeros((len(data), len(ancestry_categories)), dtype=float)
        for i, cat in enumerate(ancestry_categories):
            ancestry_dummies[:, i] = (data['ancestry'].values == cat).astype(float)

        features = np.column_stack([age, sex, ancestry_dummies])
        feature_names = (['age', 'sex']
                         + [f'ancestry_{cat}' for cat in ancestry_categories])
        return features, ancestry_categories, feature_names

    def _tune_c_cv(self, X, y, c_candidates, n_folds=5):
        """Tune C via stratified CV. Falls back to C=1.0 if too few samples."""
        n_per_class = min(np.sum(y == 0), np.sum(y == 1))
        actual_folds = min(n_folds, n_per_class)
        if actual_folds < 2:
            print("  Warning: too few samples per class for CV; using C=1.0")
            return 1.0

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True,
                              random_state=self.seed)
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
        Run k-fold CV. Per fold: train ABMIL on non-test samples, extract
        bag-level embeddings, hstack with demographic features, tune C,
        fit L1 logistic regression, evaluate on held-out test fold.
        """
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
            self.canonicalize_genes = True

        if self.debug:
            disease_rows = metadata[metadata['label'] == 1].head(self.debug_repertoires)
            healthy_rows = metadata[metadata['label'] == 0].head(self.debug_repertoires)
            metadata = pd.concat([disease_rows, healthy_rows], ignore_index=True)
            print(f"[DEBUG] Restricted to {len(metadata)} repertoires "
                  f"({len(disease_rows)} disease, {len(healthy_rows)} healthy).")

        meta_idx = metadata.set_index('specimen_label')
        all_test_rows = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"ABMIL + DEMO FOLD {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_data = metadata[~test_mask]

            print(f"Train: {len(train_data)}    Test: {len(test_data)}")

            # ------------------------------------------------------------------
            # Train ABMIL
            # ------------------------------------------------------------------
            abmil = ABMIL(
                max_instances=self.max_instances,
                M=self.M,
                L=self.L,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                patience=self.patience,
                val_split=self.val_split,
                seed=self.seed,
                sequence_col=self.sequence_col,
                v_gene_col=self.v_gene_col,
                j_gene_col=self.j_gene_col,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed,
                indices_map=self.indices_map,
                canonicalize_genes=self.canonicalize_genes,
                use_gpu=self.use_gpu,
                dropout=self.dropout,
                max_length=self.max_length,
                embedding_dim_aa=self.embedding_dim_aa,
                embedding_dim_genes=self.embedding_dim_genes,
                kernel=self.kernel,
                conv_units=self.conv_units,
                features=self.features,
            )

            abmil.train(
                train_data['file_path'].tolist(),
                train_data['label'].tolist(),
            )

            # ------------------------------------------------------------------
            # Extract bag embeddings
            # ------------------------------------------------------------------
            print("\nExtracting bag embeddings (train)...")
            X_train_emb = np.stack([
                abmil.get_bag_embedding(fp)
                for fp in tqdm(train_data['file_path'].tolist(), leave=False)
            ])
            print("Extracting bag embeddings (test)...")
            X_test_emb = np.stack([
                abmil.get_bag_embedding(fp)
                for fp in tqdm(test_data['file_path'].tolist(), leave=False)
            ])
            n_emb = X_train_emb.shape[1]

            # ------------------------------------------------------------------
            # Demographic features (ancestry categories from training data)
            # ------------------------------------------------------------------
            ancestry_cats = sorted(train_data['ancestry'].unique().tolist())
            X_train_demo, _, demo_feature_names = self.featurize_demographics(
                train_data, ancestry_categories=ancestry_cats)
            X_test_demo, _, _ = self.featurize_demographics(
                test_data, ancestry_categories=ancestry_cats)

            emb_scaler = StandardScaler()
            X_train_emb = emb_scaler.fit_transform(X_train_emb)
            X_test_emb = emb_scaler.transform(X_test_emb)

            demo_scaler = StandardScaler()
            X_train_demo = demo_scaler.fit_transform(X_train_demo)
            X_test_demo = demo_scaler.transform(X_test_demo)

            # ------------------------------------------------------------------
            # Concatenate embeddings + demographics
            # ------------------------------------------------------------------
            X_train = np.hstack([X_train_emb, X_train_demo])
            X_test = np.hstack([X_test_emb, X_test_demo])
            y_train = train_data['label'].values
            y_test = test_data['label'].values

            print(f"\nFeatures: {n_emb} embedding + {len(demo_feature_names)} demo "
                  f"= {n_emb + len(demo_feature_names)} total")

            # ------------------------------------------------------------------
            # Tune C and fit
            # ------------------------------------------------------------------
            print("\nTuning C via stratified CV...")
            best_c = self._tune_c_cv(X_train, y_train, self.C_CANDIDATES,
                                     n_folds=self.n_cv_folds)
            print(f"  Best C: {best_c}")

            model = LogisticRegression(C=best_c, penalty='l1',
                                       solver='liblinear', max_iter=1000)
            model.fit(X_train, y_train)

            n_nonzero = int(np.sum(model.coef_.ravel() != 0))
            print(f"  Non-zero coefficients: {n_nonzero} / {n_emb + len(demo_feature_names)}")

            # ------------------------------------------------------------------
            # Evaluate
            # ------------------------------------------------------------------
            test_probs = model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, test_probs)
            aupr = average_precision_score(y_test, test_probs)
            preds = (test_probs >= 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            print(f"\n--- Fold {test_fold} Test Results ---")
            print(f"  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                  f"Balanced Acc={balanced_acc:.4f}  F1={f1:.4f}")

            coefs = model.coef_.ravel()
            demo_coefs = dict(zip(demo_feature_names, coefs[n_emb:]))
            emb_nonzero = int(np.sum(coefs[:n_emb] != 0))
            print(f"\n  Non-zero embedding features: {emb_nonzero} / {n_emb}")
            print(f"  Demographic coefficients:")
            for name, coef in demo_coefs.items():
                print(f"    {name}: {coef:+.4f}")

            id_to_row = {row['specimen_label']: row
                         for _, row in test_data.iterrows()}
            for specimen, score in zip(test_data['specimen_label'], test_probs):
                if specimen not in id_to_row:
                    continue
                row = id_to_row[specimen]
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': specimen,
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'disease_model': target_disease,
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                    'method': 'ABMIL_Demo',
                    'model_score': float(score),
                })

            fold_results.append({
                'fold': test_fold,
                'auroc': auroc, 'aupr': aupr,
                'balanced_acc': balanced_acc, 'f1': f1,
                'best_c': best_c,
                'n_emb_features': n_emb,
                'n_emb_nonzero': emb_nonzero,
                'demo_coefficients': demo_coefs,
            })

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
        print(f"  ABMIL + Demo  AUROC={overall_auroc:.4f}  AUPR={overall_aupr:.4f}  "
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ABMIL bag embeddings + Demographics disease classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--max_instances', type=int, default=10000)
    parser.add_argument('--M', type=int, default=128)
    parser.add_argument('--L', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--max_length', type=int, default=40)
    parser.add_argument('--embedding_dim_aa', type=int, default=64)
    parser.add_argument('--embedding_dim_genes', type=int, default=48)
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--features', type=str, default='full',
                        choices=['full', 'cdr3_only', 'vj_only'])
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='Inner CV folds for C tuning (default: 5)')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_repertoires', type=int, default=10)
    parser.add_argument('--ext_metadata_path', type=str, default=None)
    parser.add_argument('--ext_data_dir', type=str, default=None)
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv')
    args = parser.parse_args()

    evaluator = ABMILDemographicsEvaluator(
        max_instances=args.max_instances,
        M=args.M,
        L=args.L,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        val_split=args.val_split,
        use_gpu=not args.no_gpu,
        dropout=args.dropout,
        max_length=args.max_length,
        embedding_dim_aa=args.embedding_dim_aa,
        embedding_dim_genes=args.embedding_dim_genes,
        kernel=args.kernel,
        features=args.features,
        n_cv_folds=args.n_cv_folds,
        debug=args.debug,
        debug_repertoires=args.debug_repertoires,
    )

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
