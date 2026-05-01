"""
DeepTCR embeddings + demographics disease classification.

Trains DeepTCR_WF per cross-validation fold to produce sample-level
embeddings, then concatenates them with demographic features (age, sex,
ancestry one-hot) and trains a single L1-regularized logistic regression
to predict disease status.

This follows the same pipeline as vjgene_demographics_disease_classification,
replacing V/J gene count features with DeepTCR learned embeddings.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

_DEEPTCR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'DeepTCR'
)
sys.path.insert(0, _DEEPTCR_DIR)

from DeepTCR import DeepTCR_WF
from utils_s import Get_Train_Valid_Test_KFold
from data_processing import Process_Seq
from Layers import make_test_pred_object


class DeepTCRDemographicsEvaluator:
    """
    Concatenate DeepTCR sample-level embeddings with demographic features
    (age, sex, ancestry) and train a single L1 logistic regression per CV fold.
    """

    HEALTHY_LABEL = "Healthy/Background"
    _DEEPTCR_CLASS_HEALTHY = "Healthy"
    C_CANDIDATES = [1.0, 0.2, 0.1, 0.05, 0.03]

    def __init__(self,
                 sequence_col='cdr3_aa',
                 count_col='duplicate_count',
                 v_beta_col='v_call',
                 j_beta_col='j_call',
                 max_length=40,
                 train_val_ratio=0.9,
                 random_state=7,
                 kernel=5,
                 num_concepts=64,
                 size_of_net='small',
                 epochs_min=25,
                 epochs_max=None,
                 hinge_loss_t=0.1,
                 train_loss_min=None,
                 combine_train_valid=False,
                 batch_size=25,
                 n_jobs=4,
                 device=0,
                 results_dir='results/deeptcr_demographic',
                 indices_map=None,
                 n_cv_folds=5,
                 debug=False,
                 debug_repertoires=10):
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.v_beta_col = v_beta_col
        self.j_beta_col = j_beta_col
        self.max_length = max_length
        self.train_val_ratio = train_val_ratio
        self.random_state = random_state
        self.kernel = kernel
        self.num_concepts = num_concepts
        self.size_of_net = size_of_net
        self.epochs_min = epochs_min
        self.epochs_max = epochs_max
        self.hinge_loss_t = hinge_loss_t
        self.train_loss_min = train_loss_min
        self.combine_train_valid = combine_train_valid
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.device = device
        self.results_dir = results_dir
        self.indices_map = indices_map
        self.n_cv_folds = n_cv_folds
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
                              random_state=self.random_state)
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
    # Sequence loading
    # ------------------------------------------------------------------

    def _collect_all_data(self, metadata, target_disease):
        """Read all repertoire files into flat arrays for DeepTCR's Load_Data."""
        all_seqs, all_samples, all_cls, all_counts = [], [], [], []
        all_v_beta = [] if self.v_beta_col else None
        all_j_beta = [] if self.j_beta_col else None
        written = set()

        for _, row in metadata.iterrows():
            specimen = row['specimen_label']
            class_str = target_disease if row['label'] == 1 else self._DEEPTCR_CLASS_HEALTHY

            wanted_cols = {self.sequence_col, self.count_col}
            if self.v_beta_col:
                wanted_cols.add(self.v_beta_col)
            if self.j_beta_col:
                wanted_cols.add(self.j_beta_col)

            try:
                df = pd.read_csv(row['file_path'], sep='\t',
                                 usecols=lambda c: c in wanted_cols)
            except Exception as e:
                print(f"  Warning: could not read {row['file_path']}: {e}")
                continue

            if self.sequence_col not in df.columns:
                print(f"  Warning: '{self.sequence_col}' not in {row['file_path']}, skipping.")
                continue

            if self.indices_map is not None and specimen in self.indices_map:
                df = df.iloc[self.indices_map[specimen]]

            if self.count_col not in df.columns:
                df[self.count_col] = 1

            keep_cols = [self.sequence_col, self.count_col] + [
                c for c in [self.v_beta_col, self.j_beta_col] if c and c in df.columns
            ]
            df = df[keep_cols].copy()
            df[self.count_col] = pd.to_numeric(df[self.count_col], errors='coerce')
            df = df.dropna(subset=[self.count_col])
            df = df[df[self.count_col] >= 1]

            df = Process_Seq(df, self.sequence_col)
            if len(df) == 0:
                continue

            gene_cols = [c for c in [self.v_beta_col, self.j_beta_col]
                         if c and c in df.columns]
            agg_dict = {self.count_col: 'sum'}
            for gc in gene_cols:
                agg_dict[gc] = 'first'
            df = (df.groupby(self.sequence_col, as_index=False)
                    .agg(agg_dict)
                    .sort_values(self.count_col, ascending=False)
                    .reset_index(drop=True))

            df = df[df[self.sequence_col].str.len() <= self.max_length]
            if len(df) == 0:
                continue

            n = len(df)
            all_seqs.extend(df[self.sequence_col].tolist())
            all_samples.extend([specimen] * n)
            all_cls.extend([class_str] * n)
            all_counts.extend(df[self.count_col].astype(int).tolist())

            if all_v_beta is not None:
                col = self.v_beta_col
                vals = df[col].tolist() if col in df.columns else [None] * n
                if self.canonicalize_genes:
                    from utils.gene_harmonization import canonicalize_gene
                    vals = [canonicalize_gene(v) if isinstance(v, str) else v for v in vals]
                all_v_beta.extend(vals)
            if all_j_beta is not None:
                col = self.j_beta_col
                vals = df[col].tolist() if col in df.columns else [None] * n
                if self.canonicalize_genes:
                    from utils.gene_harmonization import canonicalize_gene
                    vals = [canonicalize_gene(v) if isinstance(v, str) else v for v in vals]
                all_j_beta.extend(vals)
            written.add(specimen)

        if not all_seqs:
            return None, None, None, None, None, None, written

        return (
            np.array(all_seqs),
            np.array(all_samples),
            np.array(all_cls),
            np.array(all_counts, dtype=int),
            np.array(all_v_beta, dtype=object) if all_v_beta is not None else None,
            np.array(all_j_beta, dtype=object) if all_j_beta is not None else None,
            written,
        )

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
        Run k-fold CV. Per fold: train DeepTCR on non-test samples, extract
        sample-level embeddings, hstack with demographic features, tune C,
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

        print("\nLoading all repertoire files into memory...")
        beta_sequences, sample_labels, class_labels, counts, v_beta, j_beta, written = \
            self._collect_all_data(metadata, target_disease)

        if beta_sequences is None:
            print("Error: no sequences could be loaded.")
            return pd.DataFrame()

        print(f"Loaded {len(beta_sequences):,} sequences from {len(written)} specimens.")

        meta_idx = metadata.set_index('specimen_label')
        all_test_rows = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"DeepTCR + DEMO FOLD {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_val_data = metadata[~test_mask]

            print(f"Train+Val: {len(train_val_data)}    Test: {len(test_data)}")

            # ------------------------------------------------------------------
            # Train DeepTCR
            # ------------------------------------------------------------------
            fold_name = os.path.join(self.results_dir,
                                     f"{target_disease}_fold{test_fold}")
            os.makedirs(fold_name, exist_ok=True)

            dtcr = DeepTCR_WF(fold_name, max_length=self.max_length,
                               device=self.device)
            dtcr.Load_Data(
                beta_sequences=beta_sequences,
                sample_labels=sample_labels,
                class_labels=class_labels,
                counts=counts,
                v_beta=v_beta,
                j_beta=j_beta,
            )

            sample_name_to_idx = {s: i for i, s in enumerate(dtcr.sample_list)}
            Y = np.vstack([
                dtcr.Y[np.where(dtcr.sample_id == s)[0][0]]
                for s in dtcr.sample_list
            ])
            Vars = [np.asarray(dtcr.sample_list)]

            disease_class_idx = int(
                np.where(dtcr.lb.classes_ == target_disease)[0][0]
            )

            def _get_indices(specimens):
                return np.array(
                    [sample_name_to_idx[s]
                     for s in specimens['specimen_label']
                     if s in sample_name_to_idx],
                    dtype=int,
                )

            test_idx = _get_indices(test_data)
            train_val_idx = _get_indices(train_val_data)

            if len(test_idx) == 0:
                print(f"Warning: no test samples found for fold {test_fold}, skipping.")
                continue

            train_idx, val_idx = train_test_split(
                train_val_idx,
                train_size=self.train_val_ratio,
                random_state=self.random_state,
                stratify=Y[train_val_idx].argmax(axis=1),
            )

            dtcr.train, dtcr.valid, dtcr.test = Get_Train_Valid_Test_KFold(
                Vars=Vars, train_idx=train_idx, valid_idx=val_idx,
                test_idx=test_idx, Y=Y,
            )
            dtcr.LOO = None

            if self.combine_train_valid:
                for i in range(len(dtcr.train)):
                    dtcr.train[i] = np.concatenate(
                        (dtcr.train[i], dtcr.valid[i]), axis=0)
                    dtcr.valid[i] = dtcr.test[i]

            dtcr._reset_models()
            dtcr._build(
                kernel=self.kernel,
                num_concepts=self.num_concepts,
                size_of_net=self.size_of_net,
                epochs_min=self.epochs_min,
                epochs_max=self.epochs_max,
                hinge_loss_t=self.hinge_loss_t,
                train_loss_min=self.train_loss_min if self.combine_train_valid else None,
                convergence='training' if self.combine_train_valid else 'validation',
                batch_size=self.batch_size,
                suppress_output=False,
            )
            dtcr.test_pred = make_test_pred_object()
            dtcr._train(write=True, batch_seed=None, iteration=test_fold)

            # ------------------------------------------------------------------
            # Extract embeddings
            # ------------------------------------------------------------------
            dtcr.Sample_Features(set='all')
            sf = dtcr.sample_features  # DataFrame indexed by specimen label

            tv_specimen_names = np.concatenate([dtcr.train[0], dtcr.valid[0]])
            tv_labels_oh = np.concatenate([dtcr.train[1], dtcr.valid[1]], axis=0)
            tv_lbl = (tv_labels_oh.argmax(axis=1) == disease_class_idx).astype(int)

            test_specimen_names = dtcr.test[0]
            test_lbl = (dtcr.y_test.argmax(axis=1) == disease_class_idx).astype(int)

            # Keep only specimens present in both sf and metadata
            tv_specimen_names = np.array(
                [s for s in tv_specimen_names if s in sf.index and s in meta_idx.index])
            test_specimen_names = np.array(
                [s for s in test_specimen_names if s in sf.index and s in meta_idx.index])

            # Recompute labels aligned to filtered specimen lists
            tv_lbl = np.array([(meta_idx.loc[s, 'label']) for s in tv_specimen_names],
                              dtype=int)
            test_lbl = np.array([(meta_idx.loc[s, 'label']) for s in test_specimen_names],
                                dtype=int)

            X_tv_emb = sf.loc[tv_specimen_names].values.astype(float)
            X_test_emb = sf.loc[test_specimen_names].values.astype(float)
            n_emb = X_tv_emb.shape[1]

            # ------------------------------------------------------------------
            # Demographic features (ancestry categories from train+val)
            # ------------------------------------------------------------------
            tv_meta = meta_idx.loc[tv_specimen_names]
            test_meta = meta_idx.loc[test_specimen_names]

            ancestry_cats = sorted(tv_meta['ancestry'].unique().tolist())
            X_tv_demo, _, demo_feature_names = self.featurize_demographics(
                tv_meta, ancestry_categories=ancestry_cats)
            X_test_demo, _, _ = self.featurize_demographics(
                test_meta, ancestry_categories=ancestry_cats)

            emb_scaler = StandardScaler()
            X_tv_emb = emb_scaler.fit_transform(X_tv_emb)
            X_test_emb = emb_scaler.transform(X_test_emb)

            demo_scaler = StandardScaler()
            X_tv_demo = demo_scaler.fit_transform(X_tv_demo)
            X_test_demo = demo_scaler.transform(X_test_demo)

            # ------------------------------------------------------------------
            # Concatenate embeddings + demographics
            # ------------------------------------------------------------------
            X_train = np.hstack([X_tv_emb, X_tv_demo])
            X_test = np.hstack([X_test_emb, X_test_demo])

            print(f"\nFeatures: {n_emb} embedding + {len(demo_feature_names)} demo "
                  f"= {n_emb + len(demo_feature_names)} total")

            # ------------------------------------------------------------------
            # Tune C and fit
            # ------------------------------------------------------------------
            print("\nTuning C via stratified CV...")
            best_c = self._tune_c_cv(X_train, tv_lbl, self.C_CANDIDATES,
                                     n_folds=self.n_cv_folds)
            print(f"  Best C: {best_c}")

            model = LogisticRegression(C=best_c, penalty='l1',
                                       solver='liblinear', max_iter=1000)
            model.fit(X_train, tv_lbl)

            n_nonzero = int(np.sum(model.coef_.ravel() != 0))
            print(f"  Non-zero coefficients: {n_nonzero} / {n_emb + len(demo_feature_names)}")

            # ------------------------------------------------------------------
            # Evaluate
            # ------------------------------------------------------------------
            test_probs = model.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(test_lbl, test_probs)
            aupr = average_precision_score(test_lbl, test_probs)
            preds = (test_probs >= 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(test_lbl, preds)
            f1 = f1_score(test_lbl, preds)

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
            for specimen, score in zip(test_specimen_names, test_probs):
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
                    'method': 'DeepTCR_Demo',
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
        print(f"  DeepTCR + Demo  AUROC={overall_auroc:.4f}  AUPR={overall_aupr:.4f}  "
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
        description="DeepTCR embeddings + Demographics disease classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing AIRR .tsv.gz repertoire files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--results_dir', type=str, default='results/deeptcr_demographic',
                        help='Directory for DeepTCR checkpoints (default: results/deeptcr_demographic)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (default: 0)')
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--num_concepts', type=int, default=64)
    parser.add_argument('--size_of_net', type=str, default='small',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs_min', type=int, default=10)
    parser.add_argument('--epochs_max', type=int, default=None)
    parser.add_argument('--hinge_loss_t', type=float, default=0.1)
    parser.add_argument('--train_loss_min', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='Inner CV folds for C tuning (default: 5)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_repertoires', type=int, default=10)
    parser.add_argument('--ext_metadata_path', type=str, default=None)
    parser.add_argument('--ext_data_dir', type=str, default=None)
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv')
    args = parser.parse_args()

    evaluator = DeepTCRDemographicsEvaluator(
        kernel=args.kernel,
        num_concepts=args.num_concepts,
        size_of_net=args.size_of_net,
        epochs_min=args.epochs_min,
        epochs_max=args.epochs_max,
        hinge_loss_t=args.hinge_loss_t,
        train_loss_min=args.train_loss_min,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        device=args.device,
        results_dir=args.results_dir,
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
