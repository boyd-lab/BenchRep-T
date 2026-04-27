"""
External dataset evaluation for TCR-based disease classification.

Two evaluation modes are supported:

  hold_out: Train on ALL internal data (disease vs. Healthy/Background) with
            an internal train/val split for hyperparameter tuning, then
            evaluate on an external dataset.

  cv:       Run k-fold cross-validation directly on an external dataset using
            its pre-defined fold assignments. Mirrors the internal CV setup.

External data must be preprocessed to AIRR column conventions (cdr3_aa, v_call,
j_call) before use.  See external_data_process/preprocess_repertoires.py.

Supported models:
  - ensemble_regression: Gapped 4-mer + V/J gene logistic regression ensemble
  - emerson_2017: Fisher's exact test + Beta-Binomial generative model
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from models.ensemble_regression import Gapped_4mer_VJgene
from models.emerson_2017 import CMV_Immunosequencing_Model


SUPPORTED_MODELS = ['ensemble_regression', 'ensemble_regression_kmer',
                    'ensemble_regression_vj', 'emerson_2017']

# Model name → display name for output
MODEL_DISPLAY_NAMES = {
    'ensemble_regression': 'Ensemble_Regression',
    'ensemble_regression_kmer': 'Ensemble_Regression_Kmer',
    'ensemble_regression_vj': 'Ensemble_Regression_VJ',
    'emerson_2017': 'Emerson_2017',
}

# Map model names to ensemble regression submodel parameter
_ENSEMBLE_SUBMODEL_MAP = {
    'ensemble_regression': 'ensemble',
    'ensemble_regression_kmer': 'kmer_only',
    'ensemble_regression_vj': 'vj_only',
}


class ExternalEvaluator:
    """
    Train on an internal dataset and evaluate on an external dataset.

    Both internal and preprocessed external repertoire files use AIRR column
    conventions (cdr3_aa, v_call, j_call) with alleles already stripped.
    """

    # AIRR column conventions — fixed after preprocessing
    SEQUENCE_COL = 'cdr3_aa'
    V_GENE_COL = 'v_call'
    J_GENE_COL = 'j_call'

    # Internal dataset conventions
    INTERNAL_HEALTHY_LABEL = "Healthy/Background"
    _INTERNAL_DISEASE_COL = 'disease'
    _INTERNAL_PARTICIPANT_COL = 'participant_label'
    _INTERNAL_FILE_PREFIX = 'part_table_'
    _INTERNAL_FILE_SUFFIX = '.tsv.gz'

    # External file naming convention (set by preprocess_repertoires.py)
    _EXT_FILE_TEMPLATE = '{sample_name}_TCRB.tsv'

    def __init__(self, model_name='ensemble_regression',
                 # Ensemble Regression hyperparameters
                 val_split=0.2, n_cv_folds=5,
                 # Emerson 2017 hyperparameters
                 train_val_ratio=0.9,
                 p_value_candidates=None,
                 canonicalize_genes=False):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Supported: {SUPPORTED_MODELS}")
        self.model_name = model_name
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.train_val_ratio = train_val_ratio
        self.p_value_candidates = p_value_candidates or [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        self.canonicalize_genes = canonicalize_genes
        self.model = None

    # ------------------------------------------------------------------
    # Internal (training) dataset helpers
    # ------------------------------------------------------------------

    def _prepare_train_data(self, metadata_path, data_dir, target_disease):
        """Load internal metadata, filter to disease vs healthy, resolve file paths."""
        metadata = pd.read_csv(metadata_path, sep='\t')
        mask = metadata[self._INTERNAL_DISEASE_COL].isin(
            [target_disease, self.INTERNAL_HEALTHY_LABEL]
        )
        data = metadata[mask].copy()
        data['label'] = (data[self._INTERNAL_DISEASE_COL] == target_disease).astype(int)

        n_disease = (data['label'] == 1).sum()
        n_healthy = (data['label'] == 0).sum()
        print(f"Training data for '{target_disease}' classification:")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(data)}")

        data['file_path'] = data.apply(
            lambda row: os.path.join(
                data_dir,
                f"{self._INTERNAL_FILE_PREFIX}{row[self._INTERNAL_PARTICIPANT_COL]}"
                f"_{row['specimen_label']}{self._INTERNAL_FILE_SUFFIX}"
            ), axis=1
        )

        before = len(data)
        data = data[data['file_path'].apply(os.path.exists)].copy()
        missing = before - len(data)
        if missing > 0:
            print(f"  {missing} of {before} repertoire files not found; "
                  f"proceeding with {len(data)}.")

        return data

    # ------------------------------------------------------------------
    # External (test) dataset helpers
    # ------------------------------------------------------------------

    def _prepare_external_data(self, ext_metadata_path, ext_data_dir,
                               sample_col, disease_col,
                               healthy_label, disease_label,
                               file_template):
        """Load external metadata, construct file paths, assign binary labels."""
        metadata = pd.read_csv(ext_metadata_path, sep='\t')
        mask = metadata[disease_col].isin([disease_label, healthy_label])
        data = metadata[mask].copy()
        data['label'] = (data[disease_col] == disease_label).astype(int)

        n_disease = (data['label'] == 1).sum()
        n_healthy = (data['label'] == 0).sum()
        print(f"\nExternal data for '{disease_label}' vs '{healthy_label}':")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(data)}")

        data['file_path'] = data[sample_col].apply(
            lambda name: os.path.join(
                ext_data_dir,
                file_template.format(sample_name=name)
            )
        )

        before = len(data)
        data = data[data['file_path'].apply(os.path.exists)].copy()
        missing = before - len(data)
        if missing > 0:
            print(f"  {missing} of {before} repertoire files not found; "
                  f"proceeding with {len(data)}.")

        return data

    # ------------------------------------------------------------------
    # Model instantiation helpers
    # ------------------------------------------------------------------

    def _make_ensemble_regression(self, submodel):
        """Create an Ensemble Regression model with standard AIRR settings."""
        return Gapped_4mer_VJgene(
            val_split=self.val_split,
            n_cv_folds=self.n_cv_folds,
            sequence_col=self.SEQUENCE_COL,
            v_gene_col=self.V_GENE_COL,
            j_gene_col=self.J_GENE_COL,
            ignore_allele=True,
            canonicalize_genes=self.canonicalize_genes,
            submodel=submodel,
        )

    def _make_emerson_2017(self, p_value_threshold=1e-4):
        """Create an Emerson 2017 model with standard AIRR settings."""
        return CMV_Immunosequencing_Model(
            p_value_threshold=p_value_threshold,
            sequence_col=self.SEQUENCE_COL,
            v_col=self.V_GENE_COL,
            j_col=self.J_GENE_COL,
            ignore_allele=True,
            canonicalize_genes=self.canonicalize_genes,
        )

    # ------------------------------------------------------------------
    # Model-specific training
    # ------------------------------------------------------------------

    def _train_ensemble_regression(self, train_files, train_labels):
        """Train the Ensemble Regression model (handles internal 80/20 split)."""
        submodel = _ENSEMBLE_SUBMODEL_MAP.get(self.model_name, 'ensemble')
        self.model = self._make_ensemble_regression(submodel)
        return self.model.train(train_files, train_labels)

    def _train_emerson_2017(self, train_files, train_labels, random_state=7):
        """
        Train the Emerson 2017 model with p-value threshold tuning.

        Splits training data into train/val, tunes p_value_threshold on val,
        then retrains with the best threshold on all training data.
        """
        indices = np.arange(len(train_files))
        train_idx, val_idx = train_test_split(
            indices,
            train_size=self.train_val_ratio,
            random_state=random_state,
            stratify=train_labels,
        )

        base_files = [train_files[i] for i in train_idx]
        base_labels = [train_labels[i] for i in train_idx]
        val_files = [train_files[i] for i in val_idx]
        val_labels = [train_labels[i] for i in val_idx]

        print(f"  Train/Val split: {len(base_files)} train, {len(val_files)} val")

        # Preload all repertoires once and compute TCR statistics on the train split
        base_model = self._make_emerson_2017()
        base_model.preload_repertoires(train_files)
        base_model.compute_tcr_statistics(base_files, base_labels)

        # Tune p-value threshold on validation set
        print(f"\n--- Tuning p-value threshold ---")
        print(f"Candidates: {self.p_value_candidates}")
        tuning_results = []

        for p_val in self.p_value_candidates:
            model = self._make_emerson_2017(p_value_threshold=p_val)
            model._repertoire_cache = base_model._repertoire_cache
            model._tcr_stats_cache = base_model._tcr_stats_cache

            model.select_diagnostic_tcrs_from_cache(p_val)

            if len(model.diagnostic_tcrs) == 0:
                print(f"  p={p_val:.0e}: No diagnostic TCRs found, skipping...")
                tuning_results.append({
                    'p_value': p_val, 'n_tcrs': 0,
                    'val_auroc': 0.0, 'val_aupr': 0.0,
                })
                continue

            model.train_beta_binomial_model(base_files, base_labels)

            val_probs = [model.predict_diagnosis(fp)['probability_positive']
                         for fp in val_files]

            val_probs_arr = np.array(val_probs)
            val_labels_arr = np.array(val_labels)
            val_auroc = roc_auc_score(val_labels_arr, val_probs_arr)
            val_aupr = average_precision_score(val_labels_arr, val_probs_arr)

            tuning_results.append({
                'p_value': p_val,
                'n_tcrs': len(model.diagnostic_tcrs),
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
            })
            print(f"  p={p_val:.0e}: {len(model.diagnostic_tcrs)} TCRs, "
                  f"Val AUROC={val_auroc:.4f}, Val AUPR={val_aupr:.4f}")

        best = max(tuning_results, key=lambda x: x['val_auroc'])
        best_p_value = best['p_value']
        print(f"\nBest p-value: {best_p_value:.0e} "
              f"(Val AUROC={best['val_auroc']:.4f})")

        if best['n_tcrs'] == 0:
            print("WARNING: No diagnostic TCRs found at any threshold.")
            self.model = None
            return {
                'best_p_value': best_p_value,
                'val_auroc': 0.0,
                'val_aupr': 0.0,
                'n_diagnostic_tcrs': 0,
                'no_diagnostic_tcrs': True,
                'tuning_results': tuning_results,
            }

        # Retrain on ALL training data with the best threshold
        print(f"\nRetraining on all {len(train_files)} samples with p={best_p_value:.0e}...")
        self.model = self._make_emerson_2017(p_value_threshold=best_p_value)
        self.model._repertoire_cache = base_model._repertoire_cache
        self.model.identify_diagnostic_tcrs(train_files, train_labels)
        self.model.train_beta_binomial_model(train_files, train_labels)

        return {
            'best_p_value': best_p_value,
            'val_auroc': best['val_auroc'],
            'val_aupr': best['val_aupr'],
            'n_diagnostic_tcrs': len(self.model.diagnostic_tcrs),
            'no_diagnostic_tcrs': False,
            'tuning_results': tuning_results,
        }

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def run_external_evaluation(self,
                                train_metadata_path,
                                train_data_dir,
                                target_disease,
                                ext_metadata_path,
                                ext_data_dir,
                                ext_sample_col='sample_name',
                                ext_disease_col='disease_label',
                                ext_healthy_label='Healthy',
                                ext_disease_label='T1D',
                                file_template=None,
                                random_state=7,
                                output_csv=None):
        """
        Train on all internal data and evaluate on external dataset.

        Returns:
            Tuple of (pd.DataFrame with per-sample predictions, metrics dict).
        """
        method_name = MODEL_DISPLAY_NAMES[self.model_name]

        # --- Prepare training data ---
        train_data = self._prepare_train_data(
            train_metadata_path, train_data_dir, target_disease
        )
        train_files = train_data['file_path'].tolist()
        train_labels = train_data['label'].tolist()

        # --- Train model on all internal data ---
        print(f"\n{'='*60}")
        print(f"TRAINING {method_name} on internal dataset "
              f"({len(train_files)} samples)")
        print(f"{'='*60}")

        if self.model_name in _ENSEMBLE_SUBMODEL_MAP:
            train_result = self._train_ensemble_regression(train_files, train_labels)
        elif self.model_name == 'emerson_2017':
            train_result = self._train_emerson_2017(
                train_files, train_labels, random_state=random_state
            )
            if train_result.get('no_diagnostic_tcrs', False):
                print("Cannot evaluate: no diagnostic TCRs found.")
                return pd.DataFrame(), train_result

        # --- Prepare external data ---
        ext_data = self._prepare_external_data(
            ext_metadata_path, ext_data_dir,
            sample_col=ext_sample_col,
            disease_col=ext_disease_col,
            healthy_label=ext_healthy_label,
            disease_label=ext_disease_label,
            file_template=file_template or self._EXT_FILE_TEMPLATE,
        )
        ext_files = ext_data['file_path'].tolist()
        ext_labels = ext_data['label'].values

        # Clear cached training repertoires before predicting on external data
        self.model.clear_cache()

        # --- Predict on external data ---
        print(f"\n{'='*60}")
        print(f"EVALUATING on external dataset ({len(ext_files)} samples)")
        print(f"{'='*60}")

        ext_probs = np.array([
            self.model.predict_diagnosis(fp)['probability_positive']
            for fp in tqdm(ext_files, desc="Predicting on external data")
        ])

        # --- Compute metrics ---
        auroc = roc_auc_score(ext_labels, ext_probs)
        aupr = average_precision_score(ext_labels, ext_probs)
        ext_preds = (ext_probs >= 0.5).astype(int)
        balanced_acc = balanced_accuracy_score(ext_labels, ext_preds)
        f1 = f1_score(ext_labels, ext_preds)

        print(f"\n{'='*60}")
        print(f"EXTERNAL EVALUATION RESULTS: {method_name} on {target_disease}")
        print(f"{'='*60}")
        print(f"Training: {len(train_files)} samples (internal)")

        if self.model_name in _ENSEMBLE_SUBMODEL_MAP:
            print(f"  Best C (k-mer): {train_result['best_c_kmer']}")
            print(f"  Best C (V/J):   {train_result['best_c_vj']}")
            print(f"  Best alpha:     {train_result['best_alpha']:.1f}")
            print(f"  Val AUROC:      {train_result['val_auroc']:.4f}")
        elif self.model_name == 'emerson_2017':
            print(f"  Best p-value:       {train_result['best_p_value']:.0e}")
            print(f"  Diagnostic TCRs:    {train_result['n_diagnostic_tcrs']}")
            print(f"  Val AUROC:          {train_result['val_auroc']:.4f}")

        print(f"Evaluation: {len(ext_files)} samples (external)")
        print(f"  AUROC:        {auroc:.4f}")
        print(f"  AUPR:         {aupr:.4f}")
        print(f"  Balanced Acc: {balanced_acc:.4f}")
        print(f"  F1:           {f1:.4f}")

        # --- Build per-sample output ---
        scores_df = pd.DataFrame({
            'sample_name': ext_data[ext_sample_col].values,
            'disease_label': ext_data['label'].values,
            'disease_label_str': ext_data[ext_disease_col].values,
            'method': method_name,
            'disease_model': target_disease,
            'model_score': ext_probs,
        })

        if output_csv:
            scores_df.to_csv(output_csv, index=False)
            print(f"\nScores saved to: {output_csv}")

        return scores_df, {
            'auroc': auroc,
            'aupr': aupr,
            'balanced_acc': balanced_acc,
            'f1': f1,
            'train_result': train_result,
            'n_train': len(train_files),
            'n_external': len(ext_files),
        }

    # ------------------------------------------------------------------
    # External k-fold cross-validation
    # ------------------------------------------------------------------

    def _train_fold(self, train_files, train_labels, random_state):
        """Dispatch to the correct per-model training routine for a CV fold."""
        if self.model_name in _ENSEMBLE_SUBMODEL_MAP:
            return self._train_ensemble_regression(train_files, train_labels)
        if self.model_name == 'emerson_2017':
            return self._train_emerson_2017(
                train_files, train_labels, random_state=random_state
            )
        raise ValueError(f"Unknown model '{self.model_name}'")

    def run_cross_validation(self,
                             ext_metadata_path,
                             ext_data_dir,
                             file_template,
                             ext_sample_col='sample_name',
                             ext_disease_col='disease_label',
                             ext_healthy_label='Healthy',
                             ext_disease_label='Rheumatoid Arthritis',
                             fold_col='fold',
                             n_folds=3,
                             random_state=7,
                             target_disease=None,
                             output_csv=None):
        """
        Run k-fold cross-validation on an external dataset using its
        pre-defined fold column.

        For each fold, samples with ``fold_col == k`` form the test set and
        all other samples are passed to the model's training routine (which
        handles its own internal tuning). Predictions are aggregated across
        folds into a single per-sample scores DataFrame.

        Args:
            ext_metadata_path: Path to external metadata TSV.
            ext_data_dir: Directory with preprocessed external repertoire files.
            file_template: Format string that maps a sample name to a file
                name, e.g. ``'{sample_name}.tsv'`` or
                ``'{sample_name}_TCRB.tsv'``.
            ext_sample_col: Column in the metadata with sample identifiers.
            ext_disease_col: Column with disease labels.
            ext_healthy_label: Value in ``ext_disease_col`` that marks the
                negative class (e.g. ``'Healthy'``, ``'Controller'``).
            ext_disease_label: Value in ``ext_disease_col`` that marks the
                positive class (e.g. ``'Rheumatoid Arthritis'``,
                ``'Progressor'``).
            fold_col: Column containing pre-defined fold IDs (default: 'fold').
            n_folds: Number of folds to iterate over (default: 3).
            random_state: Random seed passed to per-model tuning.
            target_disease: Label written to the ``disease_model`` output
                column; defaults to ``ext_disease_label`` when omitted.
            output_csv: Optional path to save the per-sample scores CSV.

        Returns:
            Tuple of (pd.DataFrame with per-sample predictions, metrics dict).
        """
        method_name = MODEL_DISPLAY_NAMES[self.model_name]
        if target_disease is None:
            target_disease = ext_disease_label

        data = self._prepare_external_data(
            ext_metadata_path, ext_data_dir,
            sample_col=ext_sample_col,
            disease_col=ext_disease_col,
            healthy_label=ext_healthy_label,
            disease_label=ext_disease_label,
            file_template=file_template,
        )

        if fold_col not in data.columns:
            raise ValueError(
                f"Fold column '{fold_col}' not found in external metadata. "
                f"Available columns: {list(data.columns)}"
            )

        all_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_mask = data[fold_col] == test_fold
            test_data = data[test_mask]
            train_data = data[~test_mask]

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            if len(test_data) == 0:
                print(f"No samples in fold {test_fold}; skipping.")
                continue
            if len(train_data) == 0:
                raise ValueError(
                    f"No training samples available for fold {test_fold}."
                )

            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()
            test_files = test_data['file_path'].tolist()
            test_labels = test_data['label'].tolist()

            train_result = self._train_fold(
                train_files, train_labels, random_state=random_state
            )

            # Handle the Emerson-specific degenerate case
            if (self.model_name == 'emerson_2017'
                    and train_result.get('no_diagnostic_tcrs', False)):
                print(f"No diagnostic TCRs in fold {test_fold}; "
                      f"assigning chance-level predictions (0.5).")
                test_probs = np.full(len(test_files), 0.5)
            else:
                # Clear training repertoires before predicting on the held-out fold
                self.model.clear_cache()
                test_probs = np.array([
                    self.model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(test_files, desc="Testing")
                ])

            test_labels_arr = np.array(test_labels)
            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
            test_f1 = f1_score(test_labels_arr, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            for (_, row), score in zip(test_data.iterrows(), test_probs):
                all_rows.append({
                    'sample_name': row[ext_sample_col],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[ext_disease_col],
                    'method': method_name,
                    'disease_model': target_disease,
                    'model_score': float(score),
                    'fold': int(test_fold),
                })

            fold_results.append({
                'fold': test_fold,
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
                'train_result': train_result,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels)

        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        fold_aurocs = [r['test_auroc'] for r in fold_results]
        fold_auprs = [r['test_aupr'] for r in fold_results]
        fold_balanced_accs = [r['test_balanced_acc'] for r in fold_results]
        fold_f1s = [r['test_f1'] for r in fold_results]

        print(f"\n{'='*60}")
        print(f"OVERALL CV RESULTS: {method_name} on "
              f"{ext_disease_label} vs {ext_healthy_label}")
        print(f"{'='*60}")
        print(f"Mean Test AUROC:        {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:         {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
        print(f"Mean Test Balanced Acc: {np.mean(fold_balanced_accs):.4f} ± {np.std(fold_balanced_accs):.4f}")
        print(f"Mean Test F1:           {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"Overall AUROC (all folds combined):        {overall_auroc:.4f}")
        print(f"Overall AUPR  (all folds combined):        {overall_aupr:.4f}")
        print(f"Overall Balanced Acc (all folds combined): {overall_balanced_acc:.4f}")
        print(f"Overall F1 (all folds combined):           {overall_f1:.4f}")

        scores_df = pd.DataFrame(all_rows)
        if output_csv:
            scores_df.to_csv(output_csv, index=False)
            print(f"\nScores saved to: {output_csv}")

        return scores_df, {
            'fold_results': fold_results,
            'overall_auroc': overall_auroc,
            'overall_aupr': overall_aupr,
            'overall_balanced_acc': overall_balanced_acc,
            'overall_f1': overall_f1,
            'mean_test_auroc': float(np.mean(fold_aurocs)),
            'mean_test_aupr': float(np.mean(fold_auprs)),
            'mean_test_balanced_acc': float(np.mean(fold_balanced_accs)),
            'mean_test_f1': float(np.mean(fold_f1s)),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External dataset evaluation: hold-out (train on internal, "
                    "evaluate on external) or k-fold CV on the external dataset."
    )

    # Mode
    parser.add_argument('--mode', type=str, default='hold_out',
                        choices=['hold_out', 'cv'],
                        help='Evaluation mode: hold_out (default) trains on '
                             'internal data and evaluates on external; cv runs '
                             'k-fold CV on the external dataset.')

    # Model selection
    parser.add_argument('--model', type=str, default='ensemble_regression',
                        choices=SUPPORTED_MODELS,
                        help='Model to use (default: ensemble_regression)')

    # Internal (training) dataset — required only in hold_out mode
    parser.add_argument('--train_metadata_path', type=str, default=None,
                        help='Path to internal metadata.tsv (hold_out mode)')
    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='Directory containing internal repertoire files (hold_out mode)')
    parser.add_argument('--target_disease', type=str, default=None,
                        help='Disease label for the output scores CSV. Required '
                             'in hold_out mode (must exist in internal metadata); '
                             'optional in cv mode (defaults to --ext_disease_label).')

    # External dataset — assumes preprocessed files with AIRR column names.
    parser.add_argument('--ext_metadata_path', type=str, required=True,
                        help='Path to external metadata file')
    parser.add_argument('--ext_data_dir', type=str, required=True,
                        help='Directory containing preprocessed external repertoire files')
    parser.add_argument('--ext_sample_col', type=str, default='sample_name',
                        help='Column with sample identifiers in external metadata')
    parser.add_argument('--ext_disease_col', type=str, default='disease_label',
                        help='Column with disease labels in external metadata')
    parser.add_argument('--ext_healthy_label', type=str, default='Healthy',
                        help='Negative-class label in external metadata '
                             '(e.g. Healthy, Controller)')
    parser.add_argument('--ext_disease_label', type=str, default='T1D',
                        help='Positive-class label in external metadata '
                             '(e.g. Rheumatoid Arthritis, Progressor, T1D)')
    parser.add_argument('--file_template', type=str, default='{sample_name}_TCRB.tsv',
                        help='Format string mapping sample_name to file name. '
                             'Use "{sample_name}.tsv" for RA/TB (whose sample '
                             'names already include _TCRB or are bare), or '
                             '"{sample_name}_TCRB.tsv" for T1D.')

    # CV-mode-only args
    parser.add_argument('--fold_col', type=str, default='fold',
                        help='Fold column in external metadata (cv mode, default: fold)')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of folds to iterate over (cv mode, default: 3)')

    parser.add_argument('--canonicalize_genes', action='store_true',
                        help='Collapse Adaptive-style "-1" suffixes on IMGT '
                             'singleton TRBV families (TRBV13-1 → TRBV13). '
                             'Required when the external cohort uses Adaptive '
                             'V-gene naming alongside IMGT-named internal data.')

    # Model hyperparameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Internal val fraction for Ensemble Regression alpha tuning')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='CV folds for Ensemble Regression C tuning')
    parser.add_argument('--train_val_ratio', type=float, default=0.9,
                        help='Train/val ratio for Emerson 2017 tuning')
    parser.add_argument('--random_state', type=int, default=7,
                        help='Random seed')

    # Output
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV')

    args = parser.parse_args()

    if args.mode == 'hold_out':
        missing = [k for k in ('train_metadata_path', 'train_data_dir', 'target_disease')
                   if getattr(args, k) is None]
        if missing:
            parser.error(f"hold_out mode requires: {', '.join('--' + m for m in missing)}")

    evaluator = ExternalEvaluator(
        model_name=args.model,
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        train_val_ratio=args.train_val_ratio,
        canonicalize_genes=args.canonicalize_genes,
    )

    if args.mode == 'hold_out':
        scores_df, metrics = evaluator.run_external_evaluation(
            train_metadata_path=args.train_metadata_path,
            train_data_dir=args.train_data_dir,
            target_disease=args.target_disease,
            ext_metadata_path=args.ext_metadata_path,
            ext_data_dir=args.ext_data_dir,
            ext_sample_col=args.ext_sample_col,
            ext_disease_col=args.ext_disease_col,
            ext_healthy_label=args.ext_healthy_label,
            ext_disease_label=args.ext_disease_label,
            file_template=args.file_template,
            random_state=args.random_state,
            output_csv=args.output_csv,
        )
    else:  # cv
        scores_df, metrics = evaluator.run_cross_validation(
            ext_metadata_path=args.ext_metadata_path,
            ext_data_dir=args.ext_data_dir,
            file_template=args.file_template,
            ext_sample_col=args.ext_sample_col,
            ext_disease_col=args.ext_disease_col,
            ext_healthy_label=args.ext_healthy_label,
            ext_disease_label=args.ext_disease_label,
            fold_col=args.fold_col,
            n_folds=args.n_folds,
            random_state=args.random_state,
            target_disease=args.target_disease,
            output_csv=args.output_csv,
        )
