"""
Evaluation script for the Ensemble Regression (Gapped 4-mer + V/J gene) disease classification model.

Provides cross-validation functionality for evaluating Gapped_4mer_VJgene
on binary disease vs. Healthy/Background classification tasks.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from models.ensemble_regression import Gapped_4mer_VJgene
from utils.covariate_residualization import covariate_adjusted_predict, filter_complete_demographics


SUBMODEL_SUFFIXES = {
    'ensemble': '',
    'kmer_only': '_Kmer',
    'vj_only': '_VJ',
}


# Per-disease demographic adjustments for fair comparison. Two modes:
#
#   'filter'
#       Filter both the disease cohort and the Healthy/Background controls to
#       the same demographic slice (e.g. HIV -> African ancestry). Age bounds,
#       if given, are inclusive on both ends.
#
#   'age_match_healthy'
#       Leave the disease cohort untouched and subsample the Healthy/Background
#       controls so that their age-histogram shape matches the disease cohort.
#       ``bin_width`` controls the histogram bin size in years.
#
# Covid19 is intentionally omitted (left unadjusted).
DEMOGRAPHIC_ADJUSTMENTS = {
    'HIV': {'mode': 'filter', 'ancestry': 'African'},
    'Lupus': {'mode': 'age_match_healthy', 'bin_width': 10},
    'T1D': {'mode': 'age_match_healthy', 'bin_width': 10},
    'Influenza': {'mode': 'age_match_healthy', 'bin_width': 10},
}


class EnsembleRegressionEvaluator:
    """
    Evaluator for the Gapped 4-mer + V/J gene ensemble model.

    All hyperparameter tuning (C values via k-fold CV, ensemble alpha via
    validation sweep) is handled internally by the model's train() method,
    so the evaluator passes all non-test data directly to train().

    Set ``submodel`` to 'kmer_only' or 'vj_only' to evaluate individual
    sub-models instead of the full ensemble.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7,
                 indices_map=None, submodel='ensemble', debug=False,
                 kmer_size=4, use_gaps=True):
        """
        Args:
            val_split: Internal val fraction used by the model for alpha tuning.
            n_cv_folds: CV folds used by the model for C tuning.
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            subsample_fraction: Fraction of reads to sample per repertoire.
            subsample_seed: Random seed for reproducibility.
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
            submodel: 'ensemble' (default), 'kmer_only', or 'vj_only'.
            kmer_size: Length of k-mers to extract (default: 4).
            use_gaps: If True, include single-position gapped variants (default: True).
        """
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.indices_map = indices_map
        self.submodel = submodel
        self.kmer_size = kmer_size
        self.use_gaps = use_gaps
        self.model = None
        self.debug = debug

    def _method_name(self):
        """Build a result-CSV method label encoding submodel, k-mer size, and gap setting."""
        name = f'Ensemble_Regression_{self.kmer_size}mer'
        if self.use_gaps:
            name += '_gapped'
        name += SUBMODEL_SUFFIXES[self.submodel]
        return name
    # ------------------------------------------------------------------
    # Metadata helpers (shared pattern across evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def _apply_demographic_adjustment(self, df, target_disease):
        """
        Dispatch to the demographic adjustment strategy configured in
        ``DEMOGRAPHIC_ADJUSTMENTS`` for ``target_disease``. The input is a
        combined disease+healthy DataFrame with a ``label`` column already
        populated (1 = disease, 0 = healthy). Returns the adjusted DataFrame.
        """
        rule = DEMOGRAPHIC_ADJUSTMENTS.get(target_disease)
        if not rule:
            print(f"  No demographic adjustment defined for '{target_disease}' "
                  f"- leaving cohort unchanged.")
            return df

        mode = rule.get('mode', 'filter')
        if mode == 'filter':
            return self._apply_filter_adjustment(df, target_disease, rule)
        if mode == 'age_match_healthy':
            return self._apply_age_match_adjustment(df, target_disease, rule)
        raise ValueError(f"Unknown demographic adjustment mode '{mode}' "
                         f"for disease '{target_disease}'")

    @staticmethod
    def _apply_filter_adjustment(df, target_disease, rule):
        """Symmetric filter on both disease and healthy rows."""
        mask = pd.Series(True, index=df.index)
        desc = []
        if 'ancestry' in rule:
            mask &= (df['ancestry'] == rule['ancestry'])
            desc.append(f"ancestry={rule['ancestry']}")
        if 'sex' in rule:
            mask &= (df['sex'] == rule['sex'])
            desc.append(f"sex={rule['sex']}")
        if 'age_min' in rule or 'age_max' in rule:
            age = pd.to_numeric(df['age'], errors='coerce')
            if 'age_min' in rule:
                mask &= (age >= rule['age_min'])
            if 'age_max' in rule:
                mask &= (age <= rule['age_max'])
            desc.append(f"age in [{rule.get('age_min', '-inf')},"
                        f"{rule.get('age_max', 'inf')}]")

        before = len(df)
        filtered = df[mask].copy()
        print(f"  Demographic filter for '{target_disease}' "
              f"({', '.join(desc)}): {before} -> {len(filtered)} rows")
        return filtered

    def _apply_age_match_adjustment(self, df, target_disease, rule):
        """
        Leave the disease cohort unchanged and subsample the Healthy/Background
        controls so their age histogram (``bin_width``-year bins) has the same
        shape as the disease cohort's. The healthy sample count is maximized
        subject to not exceeding the number of available healthy samples in any
        bin; it need not equal the disease count.

        If a disease bin has no healthy samples available, that bin is dropped
        from the target distribution with a warning and the remaining disease
        bins are used to define the shape. Healthy rows with missing age are
        excluded from sampling; disease rows with missing age are kept as-is
        but cannot contribute to (or be matched in) the histogram.
        """
        bin_width = int(rule.get('bin_width', 10))

        disease_df = df[df['label'] == 1].copy()
        healthy_df = df[df['label'] == 0].copy()

        disease_age = pd.to_numeric(disease_df['age'], errors='coerce')
        healthy_age = pd.to_numeric(healthy_df['age'], errors='coerce')

        d_age_valid_mask = disease_age.notna()
        h_age_valid_mask = healthy_age.notna()

        n_disease_nan = int((~d_age_valid_mask).sum())
        n_healthy_nan = int((~h_age_valid_mask).sum())

        disease_with_age = disease_df[d_age_valid_mask]
        healthy_with_age = healthy_df[h_age_valid_mask]
        d_ages = disease_age[d_age_valid_mask]
        h_ages = healthy_age[h_age_valid_mask]

        if len(disease_with_age) == 0 or len(healthy_with_age) == 0:
            print(f"  Warning: insufficient age data to age-match "
                  f"'{target_disease}'. Leaving cohort unchanged.")
            return df

        combined_min = float(min(d_ages.min(), h_ages.min()))
        combined_max = float(max(d_ages.max(), h_ages.max()))
        bin_start = int(np.floor(combined_min / bin_width)) * bin_width
        bin_end = (int(np.floor(combined_max / bin_width)) + 1) * bin_width
        bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)
        n_bins = len(bin_edges) - 1

        d_bin = pd.cut(d_ages, bin_edges, right=False, labels=False).astype(int)
        h_bin = pd.cut(h_ages, bin_edges, right=False, labels=False).astype(int)

        d_counts = np.bincount(d_bin.values, minlength=n_bins)
        h_counts = np.bincount(h_bin.values, minlength=n_bins)

        active_bins = [i for i in range(n_bins) if d_counts[i] > 0 and h_counts[i] > 0]
        uncovered_bins = [i for i in range(n_bins) if d_counts[i] > 0 and h_counts[i] == 0]

        if uncovered_bins:
            missed = int(sum(d_counts[i] for i in uncovered_bins))
            labels = ', '.join(f"[{bin_edges[i]},{bin_edges[i+1]})"
                               for i in uncovered_bins)
            print(f"  Warning: {missed} disease sample(s) fall in bin(s) "
                  f"{labels} with no healthy counterparts; these bins are "
                  f"excluded from the target distribution.")

        if not active_bins:
            print(f"  Warning: no coverable age bins for '{target_disease}'. "
                  f"Leaving cohort unchanged.")
            return df

        active_d = np.array([d_counts[i] for i in active_bins], dtype=float)
        active_h = np.array([h_counts[i] for i in active_bins], dtype=float)
        props = active_d / active_d.sum()
        max_n = int(np.floor(float(np.min(active_h / props))))
        if max_n <= 0:
            print(f"  Warning: maximum matched healthy N is 0 for "
                  f"'{target_disease}'. Leaving cohort unchanged.")
            return df

        rng = np.random.RandomState(self.subsample_seed)
        sampled_parts = []
        h_final_counts = np.zeros(n_bins, dtype=int)
        # Map each healthy row's original index to its bin index.
        h_bin_by_index = pd.Series(h_bin.values, index=healthy_with_age.index)

        for j, bin_idx in enumerate(active_bins):
            target = int(round(max_n * props[j]))
            target = min(target, int(active_h[j]))
            if target <= 0:
                continue
            pool = healthy_with_age.loc[h_bin_by_index[h_bin_by_index == bin_idx].index]
            sampled = pool.sample(n=target, random_state=int(rng.randint(0, 2**31 - 1)))
            sampled_parts.append(sampled)
            h_final_counts[bin_idx] = target

        sampled_healthy = (pd.concat(sampled_parts, axis=0)
                           if sampled_parts else healthy_with_age.iloc[0:0])

        print(f"  Age-matched cohort for '{target_disease}' "
              f"(bin width {bin_width}y): disease {len(disease_df)} (unchanged), "
              f"healthy {len(healthy_df)} -> {len(sampled_healthy)}")
        print(f"    Per-bin (disease / healthy available / healthy sampled):")
        for i in range(n_bins):
            if d_counts[i] == 0 and h_counts[i] == 0:
                continue
            print(f"      [{bin_edges[i]:3d},{bin_edges[i+1]:3d}):  "
                  f"{d_counts[i]:3d}  /  {h_counts[i]:3d}  /  {h_final_counts[i]:3d}")
        if n_disease_nan:
            print(f"    Note: {n_disease_nan} disease row(s) have missing age "
                  f"(kept, not used for histogram)")
        if n_healthy_nan:
            print(f"    Note: {n_healthy_nan} healthy row(s) dropped for missing age")

        return pd.concat([disease_df, sampled_healthy], axis=0)

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             require_demographics=False,
                             adjust_distribution_by_demographics=False):
        """
        Filter metadata to target disease vs. Healthy/Background and add binary labels.

        Args:
            metadata: DataFrame with metadata.
            target_disease: Disease name to classify.
            disease_col: Column with disease labels.
            require_demographics: If True, drop rows with missing age, sex,
                or ancestry so the subset matches the demographic baseline.
            adjust_distribution_by_demographics: If True, apply the per-disease demographic
                filter from ``DEMOGRAPHIC_ADJUSTMENTS`` to both the disease
                cohort and the Healthy/Background controls.

        Returns:
            DataFrame with a 'label' column (1 = disease, 0 = healthy).
        """
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        if adjust_distribution_by_demographics:
            filtered = self._apply_demographic_adjustment(filtered, target_disease)

        if require_demographics:
            before = len(filtered)
            filtered = filtered.dropna(subset=['age', 'sex', 'ancestry'])
            filtered = filtered[filtered['ancestry'].str.strip() != '']
            after = len(filtered)
            if before != after:
                print(f"  Dropped {before - after} rows with missing demographics "
                      f"({before} -> {after})")

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease: {n_disease}  Healthy: {n_healthy}  Total: {len(filtered)}")
        return filtered

    def get_available_diseases(self, metadata_path, disease_col='disease'):
        metadata = self.load_metadata(metadata_path)
        return [d for d in metadata[disease_col].unique() if d != self.HEALTHY_LABEL]

    def construct_file_path(self, participant_label, specimen_label, data_dir,
                            file_prefix='part_table_', file_suffix='.tsv.gz'):
        return os.path.join(data_dir,
                            f"{file_prefix}{participant_label}_{specimen_label}{file_suffix}")

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: self.construct_file_path(
                row[participant_col], row['specimen_label'], data_dir, file_prefix, file_suffix
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
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              tune_parameters=True,
                              allowed_participants=None,
                              require_demographics=False,
                              adjust_distribution_by_demographics=False,
                              covariate_adjust=False,
                              debug_repertoires=0):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold, all non-test samples are passed to model.train() which
        handles internal hyperparameter tuning (C via CV, alpha via val sweep).

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease name to classify against Healthy/Background.
            data_dir: Directory containing repertoire .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix: Filename prefix (default: 'part_table_').
            file_suffix: Filename suffix (default: '.tsv.gz').
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined fold IDs (0, 1, 2).
            n_folds: Number of folds (default: 3).
            random_state: Random seed for reproducibility (default: 7).
            tune_parameters: Accepted for API compatibility (tuning is always
                             handled internally by the model's train() method).
            allowed_participants: Optional set of specimen labels to restrict to
                                  (e.g., for depth experiments filtering to
                                  repertoires with sufficient sequences).
            require_demographics: If True, drop repertoires with missing
                                  demographic data (age, sex, ancestry) so the
                                  subset matches the demographic baseline.
            adjust_distribution_by_demographics: If True, apply the per-disease demographic
                                 filter from ``DEMOGRAPHIC_ADJUSTMENTS`` to
                                 both the disease and healthy cohorts to make
                                 the comparison fairer.

        Returns:
            Dict with fold-level results and overall AUROC / AUPR.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col,
                                                require_demographics=require_demographics,
                                                adjust_distribution_by_demographics=adjust_distribution_by_demographics)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                        file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if debug_repertoires > 0:
            disease_rows = metadata[metadata['label'] == 1].head(debug_repertoires)
            healthy_rows = metadata[metadata['label'] == 0].head(debug_repertoires)
            metadata = pd.concat([disease_rows, healthy_rows])
            print(f"Debug: subsampled to {len(disease_rows)} disease + {len(healthy_rows)} healthy repertoires")

        # Filter to allowed participants if specified (e.g., for depth experiments)
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
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_data = metadata[~test_mask]

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()
            test_files = test_data['file_path'].tolist()
            test_labels = test_data['label'].tolist()

            # Fresh model per fold
            self.model = Gapped_4mer_VJgene(
                val_split=self.val_split,
                n_cv_folds=self.n_cv_folds,
                sequence_col=self.sequence_col,
                v_gene_col=self.v_gene_col,
                j_gene_col=self.j_gene_col,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed,
                indices_map=self.indices_map,
                submodel=self.submodel,
                kmer_size=self.kmer_size,
                use_gaps=self.use_gaps,
            )

            train_result = self.model.train(train_files, train_labels)

            if covariate_adjust:
                # ----------------------------------------------------------
                # Score-level covariate adjustment
                # ----------------------------------------------------------
                tv_cov = filter_complete_demographics(train_data)
                test_cov = filter_complete_demographics(test_data)
                print(f"  Covariate adjust: {len(tv_cov)} train, "
                      f"{len(test_cov)} test samples with complete demographics.")

                tv_scores = [
                    self.model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(tv_cov['file_path'].tolist(), desc="Scoring train")
                ]
                test_scores = [
                    self.model.predict_diagnosis(fp)['probability_positive']
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
                # ----------------------------------------------------------
                # Standard prediction
                # ----------------------------------------------------------
                test_probs = np.array([
                    self.model.predict_diagnosis(fp)['probability_positive']
                    for fp in tqdm(test_files, desc="Testing")
                ])
                test_labels_arr = np.array(test_labels)

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
                'best_c_kmer': train_result['best_c_kmer'],
                'best_c_vj': train_result['best_c_vj'],
                'best_alpha': train_result['best_alpha'],
                'val_auroc': train_result['val_auroc'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels_arr.tolist())

        # Overall metrics (all test predictions concatenated across folds)
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
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

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble Regression (Gapped 4-mer + V/J gene) Disease Classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Internal val fraction for alpha tuning (default: 0.2)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='CV folds for C tuning (default: 5)')
    parser.add_argument('--submodel', type=str, default='ensemble',
                        choices=['ensemble', 'kmer_only', 'vj_only'],
                        help='Sub-model to evaluate: ensemble (default), '
                             'kmer_only (gapped 4-mer only), or vj_only (V/J gene only)')
    parser.add_argument('--require_demographics', action='store_true',
                        help='Drop repertoires with missing demographic data '
                             '(age, sex, ancestry) to match demographic baseline subset')
    parser.add_argument('--adjust_distribution_by_demographics', action='store_true',
                        help='Apply per-disease cohort distribution adjustment for fair '
                             'comparison. HIV: filter both cohorts to African ancestry. '
                             'Lupus/T1D/Influenza: keep the disease cohort unchanged and '
                             'subsample Healthy/Background so its age distribution (10y bins) '
                             'matches the disease cohort. Covid19 is left unadjusted.')
    parser.add_argument('--covariate_adjust', action='store_true',
                        help='Residualize model scores against demographics (age, sex, ancestry) '
                             'and train an L1 logistic regression head (requires complete demographics)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output and no file existence filtering')
    parser.add_argument('--debug_repertoires', type=int, default=0,
                        help='Subsample to N disease + N healthy repertoires for fast debug runs (0 = disabled)')
    parser.add_argument('--kmer_size', type=int, default=4,
                        help='Length of k-mers to extract from CDR3 sequences (default: 4)')
    parser.add_argument('--no_gaps', action='store_true',
                        help='Disable single-position gapped k-mer variants; '
                             'extract plain k-mers only')
    args = parser.parse_args()

    evaluator = EnsembleRegressionEvaluator(
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        submodel=args.submodel,
        debug=args.debug,
        kmer_size=args.kmer_size,
        use_gaps=not args.no_gaps,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        require_demographics=args.require_demographics,
        adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
        covariate_adjust=args.covariate_adjust,
        debug_repertoires=args.debug_repertoires,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
