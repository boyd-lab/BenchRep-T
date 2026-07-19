"""
Evaluation script for Emerson 2017 CMV classification model.

This module provides cross-validation and parameter tuning functionality
for evaluating the CMV_Immunosequencing_Model on disease classification tasks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from models.emerson_2017 import CMV_Immunosequencing_Model
from utils.covariate_residualization import covariate_adjusted_predict, filter_complete_demographics
from utils.cohort_adjustments import apply_cohort_adjustment
from utils.outer_fold import outer_test_folds, split_metadata


class Emerson2017Evaluator:
    """
    Evaluator for the Emerson 2017 CMV Immunosequencing Model.
    
    Provides functionality for:
    - Loading metadata with fold assignments
    - Creating binary labels for disease vs healthy classification
    - Hyperparameter tuning using validation sets
    - K-fold cross-validation with pre-defined folds
    """
    
    # Constants for label values
    HEALTHY_LABEL = "Healthy/Background"
    
    def __init__(self, train_val_ratio=0.9, p_value_threshold=1e-4, sequence_col='cdr3_aa',
                 v_col='v_call', j_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None):
        """
        Initialize the evaluator.

        Args:
            train_val_ratio: Ratio of training data in train/val split (default: 0.9, i.e., 9:1)
            p_value_threshold: Initial p-value threshold for Fisher's exact test (default: 1e-4)
            sequence_col: Column name containing CDR3 amino acid sequences (default: 'cdr3_aa')
            v_col: Column name containing V gene calls (default: 'v_call')
            j_col: Column name containing J gene calls (default: 'j_call')
            subsample_fraction: Fraction of reads to keep for depth simulation (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling (default: 42)
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set)
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
        """
        self.train_val_ratio = train_val_ratio
        self.p_value_threshold = p_value_threshold
        self.sequence_col = sequence_col
        self.v_col = v_col
        self.j_col = j_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.canonicalize_genes = False
        self.model = None
    
    def load_metadata(self, metadata_path):
        """
        Load metadata file that maps repertoire files to folds.
        
        Args:
            metadata_path: Path to metadata.tsv file with columns for file paths, 
                           labels, and fold assignments.
        
        Returns:
            DataFrame with metadata
        """
        metadata = pd.read_csv(metadata_path, sep='\t')
        return metadata
    
    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             adjust_distribution_by_demographics=False,
                             random_baseline=False,
                             random_baseline_seed=7):
        """
        Prepare binary classification data for a specific disease.

        Filters the metadata to include only:
        - Samples with the target disease (label = 1)
        - Healthy/Background samples (label = 0)

        Args:
            metadata: DataFrame with metadata
            target_disease: Name of the disease to classify (e.g., 'Lupus', 'T1D', 'HIV')
            disease_col: Column name containing disease labels (default: 'disease')
            adjust_distribution_by_demographics: If True, apply the per-disease
                demographic cohort adjustment from ``DEMOGRAPHIC_ADJUSTMENTS``.
            random_baseline: If True (with ``adjust_distribution_by_demographics``),
                keep the disease side identical to the demographic-matched run
                but resample healthy uniformly at random to the same target N.
            random_baseline_seed: RNG seed for the random-baseline draw.

        Returns:
            DataFrame with filtered data and a new 'label' column (1 for disease, 0 for healthy)
        """
        # Filter to include only target disease and healthy samples
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered_data = metadata[mask].copy()

        # Create binary labels: 1 for disease, 0 for healthy
        filtered_data['label'] = (filtered_data[disease_col] == target_disease).astype(int)

        if adjust_distribution_by_demographics:
            filtered_data = apply_cohort_adjustment(
                filtered_data, target_disease,
                seed=random_baseline_seed if random_baseline else self.subsample_seed,
                random_baseline=random_baseline,
            )

        n_disease = (filtered_data['label'] == 1).sum()
        n_healthy = (filtered_data['label'] == 0).sum()

        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {len(filtered_data)} samples")

        return filtered_data
    
    def get_available_diseases(self, metadata_path, disease_col='disease'):
        """
        List all available diseases in the metadata (excluding Healthy/Background).
        
        Args:
            metadata_path: Path to metadata.tsv file
            disease_col: Column name containing disease labels
            
        Returns:
            List of disease names
        """
        metadata = self.load_metadata(metadata_path)
        diseases = metadata[disease_col].unique().tolist()
        diseases = [d for d in diseases if d != self.HEALTHY_LABEL]
        return diseases
    
    def construct_file_path(self, participant_label, specimen_label, data_dir,
                            file_prefix='part_table_', file_suffix='.tsv.gz'):
        """
        Construct the full file path from a participant and specimen label.

        Args:
            participant_label: The participant ID (e.g., 'BFI-0003052')
            specimen_label: The specimen ID (e.g., 'S001')
            data_dir: Root directory containing the data files
            file_prefix: Prefix to add before participant label (default: 'part_table_')
            file_suffix: Suffix to add after specimen label (default: '.tsv.gz')

        Returns:
            Full file path (e.g., '/path/to/data/part_table_BFI-0003052_S001.tsv.gz')
        """
        filename = f"{file_prefix}{participant_label}_{specimen_label}{file_suffix}"
        return os.path.join(data_dir, filename)

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        """
        Add a 'file_path' column to metadata by constructing paths from participant and
        specimen labels.

        Args:
            metadata: DataFrame with metadata
            data_dir: Root directory containing the data files
            participant_col: Column name containing participant labels (default: 'participant_label')
            file_prefix: Prefix to add before participant label (default: 'part_table_')
            file_suffix: Suffix to add after specimen label (default: '.tsv.gz')

        Returns:
            DataFrame with added 'file_path' column
        """
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: self.construct_file_path(
                row[participant_col], row['specimen_label'], data_dir, file_prefix, file_suffix
            ), axis=1
        )
        return metadata
    
    def filter_existing_files(self, metadata):
        """
        Filter metadata to only include rows where the file_path exists.
        
        Args:
            metadata: DataFrame with 'file_path' column
        
        Returns:
            DataFrame filtered to only include existing files
        """
        original_count = len(metadata)
        
        # Check which files exist
        metadata = metadata.copy()
        metadata['file_exists'] = metadata['file_path'].apply(os.path.exists)
        
        # Filter to existing files only
        filtered_metadata = metadata[metadata['file_exists']].drop(columns=['file_exists'])
        
        filtered_count = len(filtered_metadata)
        missing_count = original_count - filtered_count
        
        if missing_count > 0:
            print(f"Note: {missing_count} of {original_count} files not found in directory. "
                  f"Proceeding with {filtered_count} available files.")
        
        return filtered_metadata

    def tune_and_train(self, train_files, train_labels, val_files, val_labels,
                       p_value_candidates=None):
        """
        Tune p_value_threshold using validation set, then train with optimal threshold.
        
        Uses optimized caching:
        1. Preloads all repertoire files once
        2. Computes TCR statistics (counts + p-values) once
        3. For each p-value threshold, just filters the precomputed p-values
        
        Args:
            train_files: List of training file paths
            train_labels: List of training labels
            val_files: List of validation file paths
            val_labels: List of validation labels
            p_value_candidates: List of p-value thresholds to try 
                               (default: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        
        Returns:
            Dictionary with tuning results and best parameters
        """
        if p_value_candidates is None:
            p_value_candidates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        
        print("--- Parameter Tuning ---")
        print(f"Testing p-value thresholds: {p_value_candidates}")
        
        # Create a base model instance for preloading and caching
        base_model = CMV_Immunosequencing_Model(
            p_value_threshold=p_value_candidates[0], sequence_col=self.sequence_col,
            v_col=self.v_col, j_col=self.j_col,
            subsample_fraction=self.subsample_fraction, subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n, indices_map=self.indices_map,
            canonicalize_genes=self.canonicalize_genes,
        )
        
        # Step 1: Preload all repertoire files once (train + val)
        all_files = list(train_files) + list(val_files)
        base_model.preload_repertoires(all_files)
        
        # Step 2: Compute TCR statistics once (counts + p-values for all TCRs)
        base_model.compute_tcr_statistics(train_files, train_labels)
        
        tuning_results = []
        
        # Step 3: For each p-value, just filter the cached statistics (very fast)
        print("\n--- Evaluating p-value thresholds ---")
        for p_val in p_value_candidates:
            # Create new model sharing all caches
            model = CMV_Immunosequencing_Model(
                p_value_threshold=p_val, sequence_col=self.sequence_col,
                v_col=self.v_col, j_col=self.j_col,
                subsample_fraction=self.subsample_fraction, subsample_seed=self.subsample_seed,
                indices_map=self.indices_map,
                canonicalize_genes=self.canonicalize_genes,
            )
            model._repertoire_cache = base_model._repertoire_cache
            model._tcr_stats_cache = base_model._tcr_stats_cache
            
            # Select diagnostic TCRs by filtering cached p-values (instant)
            model.select_diagnostic_tcrs_from_cache(p_val)
            
            # Skip if no diagnostic TCRs found
            if len(model.diagnostic_tcrs) == 0:
                print(f"  p={p_val:.0e}: No diagnostic TCRs found, skipping...")
                tuning_results.append({
                    'p_value_threshold': p_val,
                    'n_diagnostic_tcrs': 0,
                    'val_auroc': 0.0,
                    'val_aupr': 0.0,
                    'val_balanced_acc': 0.0,
                    'val_f1': 0.0,
                })
                continue
            
            # Train Beta-Binomial model (uses cached repertoires)
            model.train_beta_binomial_model(train_files, train_labels)
            
            # Evaluate on validation set (uses cached repertoires)
            val_probs = []
            for file_path in val_files:
                result = model.predict_diagnosis(file_path)
                val_probs.append(result['probability_positive'])
            
            val_probs = np.array(val_probs)
            val_labels_arr = np.array(val_labels)
            
            # Compute AUROC, AUPR, Balanced Accuracy, and F1
            val_auroc = roc_auc_score(val_labels_arr, val_probs)
            val_aupr = average_precision_score(val_labels_arr, val_probs)
            val_preds = (val_probs >= 0.5).astype(int)
            val_balanced_acc = balanced_accuracy_score(val_labels_arr, val_preds)
            val_f1 = f1_score(val_labels_arr, val_preds)

            tuning_results.append({
                'p_value_threshold': p_val,
                'n_diagnostic_tcrs': len(model.diagnostic_tcrs),
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
                'val_balanced_acc': val_balanced_acc,
                'val_f1': val_f1,
            })

            print(f"  p={p_val:.0e}: {len(model.diagnostic_tcrs)} TCRs, Val AUROC={val_auroc:.4f}, "
                  f"Val AUPR={val_aupr:.4f}, Balanced Acc={val_balanced_acc:.4f}, F1={val_f1:.4f}")
        
        # Find best threshold (using AUROC as primary metric)
        best_result = max(tuning_results, key=lambda x: x['val_auroc'])
        best_p_value = best_result['p_value_threshold']

        print(f"\nBest p-value threshold: {best_p_value:.0e} "
              f"(Val AUROC={best_result['val_auroc']:.4f}, Val AUPR={best_result['val_aupr']:.4f}, "
              f"Balanced Acc={best_result['val_balanced_acc']:.4f}, F1={best_result['val_f1']:.4f})")

        # If no threshold found any diagnostic TCRs, signal failure
        if best_result['n_diagnostic_tcrs'] == 0:
            print("WARNING: No diagnostic TCRs found at any threshold. "
                  "Model will output random-chance predictions for this fold.")
            self.model = None
            return {
                'tuning_results': tuning_results,
                'best_p_value_threshold': best_p_value,
                'best_val_auroc': 0.0,
                'best_val_aupr': 0.0,
                'best_val_balanced_acc': 0.0,
                'best_val_f1': 0.0,
                'best_n_diagnostic_tcrs': 0,
                'no_diagnostic_tcrs': True
            }

        # Final model with best threshold (reuses all caches)
        self.model = CMV_Immunosequencing_Model(
            p_value_threshold=best_p_value, sequence_col=self.sequence_col,
            v_col=self.v_col, j_col=self.j_col,
            subsample_fraction=self.subsample_fraction, subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n, indices_map=self.indices_map,
            canonicalize_genes=self.canonicalize_genes,
        )
        self.model._repertoire_cache = base_model._repertoire_cache
        self.model._tcr_stats_cache = base_model._tcr_stats_cache
        self.model.select_diagnostic_tcrs_from_cache(best_p_value)
        self.model.train_beta_binomial_model(train_files, train_labels)

        return {
            'tuning_results': tuning_results,
            'best_p_value_threshold': best_p_value,
            'best_val_auroc': best_result['val_auroc'],
            'best_val_aupr': best_result['val_aupr'],
            'best_val_balanced_acc': best_result['val_balanced_acc'],
            'best_val_f1': best_result['val_f1'],
            'best_n_diagnostic_tcrs': len(self.model.diagnostic_tcrs),
            'no_diagnostic_tcrs': False
        }
    
    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              tune_parameters=True, p_value_candidates=None,
                              allowed_participants=None,
                              adjust_distribution_by_demographics=False,
                              random_baseline=False,
                              random_baseline_seed=7,
                              covariate_adjust=False,
                              model_save_dir=None,
                              ext_metadata_path=None, ext_data_dir=None,
                              ext_file_template='{participant_label}_TCRB.tsv'):
        """
        Run k-fold cross-validation using pre-defined fold assignments.
        
        Args:
            metadata_path: Path to metadata.tsv file
            target_disease: Name of the disease to classify (e.g., 'Lupus', 'T1D', 'HIV')
            data_dir: Root directory containing the repertoire data files
            participant_col: Column name containing participant labels (default: 'participant_label')
            file_prefix: Prefix for file names (default: 'part_table_')
            file_suffix: Suffix for file names (default: '.tsv.gz')
            disease_col: Column name containing disease labels (default: 'disease')
            fold_col: Column name containing fold assignments 
                     (default: 'malid_cross_validation_fold_id_when_in_test_set')
            n_folds: Number of folds (default: 3)
            random_state: Random seed for train/val split reproducibility
            tune_parameters: Whether to tune p_value_threshold using validation set (default: True)
            p_value_candidates: List of p-value thresholds to try during tuning
                               (default: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        
        Returns:
            Dictionary containing results for each fold and overall metrics
        """
        # Load and prepare data with binary labels
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(
            raw_metadata, target_disease, disease_col,
            adjust_distribution_by_demographics=adjust_distribution_by_demographics,
            random_baseline=random_baseline,
            random_baseline_seed=random_baseline_seed,
        )

        # Add file paths
        metadata = self.add_file_paths(
            metadata, data_dir, participant_col, file_prefix, file_suffix
        )
        
        # Filter to only include files that exist
        metadata = self.filter_existing_files(metadata)

        # Optional: merge an external cohort by fold (canonicalizes V/J calls).
        if ext_metadata_path is not None:
            from utils.cohort_merge import prepare_merged_cohort
            metadata = prepare_merged_cohort(
                metadata, ext_metadata_path, ext_data_dir, target_disease,
                ext_file_template=ext_file_template,
                healthy_label=self.HEALTHY_LABEL,
                fold_col=fold_col, disease_col=disease_col,
            )
            self.canonicalize_genes = True

        # Filter to allowed participants if specified (e.g., for min-sequence-count filtering)
        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants set.")

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in outer_test_folds(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            # Split data by fold
            train_val_data, test_data = split_metadata(
                metadata, fold_col, test_fold)
            train_data, val_data = train_test_split(
                train_val_data, train_size=self.train_val_ratio,
                random_state=random_state, stratify=train_val_data['label'])

            print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

            # Extract file paths and labels (using constructed 'file_path' and binary 'label' columns)
            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()

            val_files = val_data['file_path'].tolist()
            val_labels = val_data['label'].tolist()

            test_files = test_data['file_path'].tolist()
            test_labels = test_data['label'].tolist()

            # Train with or without parameter tuning
            if tune_parameters:
                # Use validation set to tune p_value_threshold
                tuning_result = self.tune_and_train(
                    train_files, train_labels,
                    val_files, val_labels,
                    p_value_candidates=p_value_candidates
                )
                best_p_value = tuning_result['best_p_value_threshold']
                val_auroc = tuning_result['best_val_auroc']
                val_aupr = tuning_result['best_val_aupr']
                val_balanced_acc = tuning_result['best_val_balanced_acc']
                val_f1 = tuning_result['best_val_f1']

                # Handle case where no diagnostic TCRs were found
                if tuning_result.get('no_diagnostic_tcrs', False):
                    print(f"No diagnostic TCRs found for fold {test_fold}. "
                          f"Assigning chance-level predictions (0.5).")
                    test_probs = np.full(len(test_files), 0.5)
                    test_labels_arr = np.array(test_labels)
                    test_auroc = roc_auc_score(test_labels_arr, test_probs)
                    test_aupr = average_precision_score(test_labels_arr, test_probs)
                    test_preds = (test_probs >= 0.5).astype(int)
                    test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
                    test_f1 = f1_score(test_labels_arr, test_preds)
                    print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                          f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

                    method_name = 'Emerson_2017'
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

                    fold_results.append({
                        'fold': test_fold,
                        'best_p_value_threshold': best_p_value,
                        'val_auroc': val_auroc,
                        'val_aupr': val_aupr,
                        'test_auroc': test_auroc,
                        'test_aupr': test_aupr,
                        'test_balanced_acc': test_balanced_acc,
                        'test_f1': test_f1,
                    })

                    all_probs.extend(test_probs.tolist())
                    all_labels.extend(test_labels)
                    continue
            else:
                # Train without tuning (use initial p_value_threshold)
                self.model = CMV_Immunosequencing_Model(
                    p_value_threshold=self.p_value_threshold, sequence_col=self.sequence_col,
                    v_col=self.v_col, j_col=self.j_col,
                    subsample_fraction=self.subsample_fraction, subsample_seed=self.subsample_seed,
                    indices_map=self.indices_map,
                    canonicalize_genes=self.canonicalize_genes,
                )
                self.model.identify_diagnostic_tcrs(train_files, train_labels)
                self.model.train_beta_binomial_model(train_files, train_labels)

                # Evaluate on validation set
                val_probs = []
                for file_path in tqdm(val_files, desc="Validating", leave=False):
                    result = self.model.predict_diagnosis(file_path)
                    val_probs.append(result['probability_positive'])
                val_probs = np.array(val_probs)
                val_labels_arr = np.array(val_labels)
                val_auroc = roc_auc_score(val_labels_arr, val_probs)
                val_aupr = average_precision_score(val_labels_arr, val_probs)
                val_preds = (val_probs >= 0.5).astype(int)
                val_balanced_acc = balanced_accuracy_score(val_labels_arr, val_preds)
                val_f1 = f1_score(val_labels_arr, val_preds)
                best_p_value = self.p_value_threshold
                tuning_result = None

            print(f"\nFinal Validation AUROC: {val_auroc:.4f}, AUPR: {val_aupr:.4f}, "
                  f"Balanced Acc: {val_balanced_acc:.4f}, F1: {val_f1:.4f}")

            if model_save_dir is not None and self.model is not None:
                self.model.save(os.path.join(model_save_dir, target_disease,
                                             f'fold{test_fold}', 'model.pkl'))

            if covariate_adjust:
                # ----------------------------------------------------------
                # Diagnostic-TCR-presence covariate adjustment
                # ----------------------------------------------------------
                train_val_combined = pd.concat([train_data, val_data], ignore_index=True)
                tv_cov = filter_complete_demographics(train_val_combined)
                test_cov = filter_complete_demographics(test_data)
                print(f"  Covariate adjust: {len(tv_cov)} train/val, "
                      f"{len(test_cov)} test samples with complete demographics.")

                n_diag = len(self.model.diagnostic_tcr_list)
                if n_diag == 0:
                    # Fisher's test produced no diagnostic TCRs; fall back to
                    # the (n, 1) score-level adjustment so the fold doesn't crash.
                    print("  WARNING: 0 diagnostic TCRs; falling back to score-level adjustment.")
                    tv_scores = [
                        self.model.predict_diagnosis(fp)['probability_positive']
                        for fp in tqdm(tv_cov['file_path'].tolist(), desc="Scoring train/val")
                    ]
                    test_scores = [
                        self.model.predict_diagnosis(fp)['probability_positive']
                        for fp in tqdm(test_cov['file_path'].tolist(), desc="Scoring test")
                    ]
                    X_tv = np.array(tv_scores).reshape(-1, 1)
                    X_test_emb = np.array(test_scores).reshape(-1, 1)
                else:
                    tv_presence = [
                        self.model.get_diagnostic_tcr_presence(fp)
                        for fp in tqdm(tv_cov['file_path'].tolist(),
                                       desc="Building train/val presence matrix")
                    ]
                    test_presence = [
                        self.model.get_diagnostic_tcr_presence(fp)
                        for fp in tqdm(test_cov['file_path'].tolist(),
                                       desc="Building test presence matrix")
                    ]
                    X_tv = np.vstack(tv_presence).astype(np.float32)
                    X_test_emb = np.vstack(test_presence).astype(np.float32)
                    print(f"  Presence matrix: train/val {X_tv.shape}, test {X_test_emb.shape}")

                test_probs = covariate_adjusted_predict(
                    X_tv, tv_cov, tv_cov['label'].values, X_test_emb, test_cov
                )
                test_labels_arr = test_cov['label'].values

                method_name = 'Emerson_2017_CovAdj'
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
                # Standard Emerson 2017 prediction
                # ----------------------------------------------------------
                test_probs = []
                for file_path in tqdm(test_files, desc="Testing"):
                    result = self.model.predict_diagnosis(file_path)
                    test_probs.append(result['probability_positive'])

                test_probs = np.array(test_probs)
                test_labels_arr = np.array(test_labels)

                method_name = 'Emerson_2017'
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

            # Compute AUROC, AUPR, Balanced Accuracy, and F1 for test set
            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            test_preds = (test_probs >= 0.5).astype(int)
            test_balanced_acc = balanced_accuracy_score(test_labels_arr, test_preds)
            test_f1 = f1_score(test_labels_arr, test_preds)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}, "
                  f"Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")

            fold_results.append({
                'fold': test_fold,
                'best_p_value_threshold': best_p_value,
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_balanced_acc': test_balanced_acc,
                'test_f1': test_f1,
            })

            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels_arr.tolist())

        # Calculate overall metrics across all folds
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)
        overall_preds = (all_probs_arr >= 0.5).astype(int)
        overall_balanced_acc = balanced_accuracy_score(all_labels_arr, overall_preds)
        overall_f1 = f1_score(all_labels_arr, overall_preds)

        print(f"\n{'='*60}")
        print(f"OVERALL CROSS-VALIDATION RESULTS: {target_disease} vs Healthy")
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
        print(f"Overall AUPR (all folds combined):         {overall_aupr:.4f}")
        print(f"Overall Balanced Acc (all folds combined): {overall_balanced_acc:.4f}")
        print(f"Overall F1 (all folds combined):           {overall_f1:.4f}")

        if tune_parameters:
            print(f"\nBest p-value thresholds per fold:")
            for r in fold_results:
                print(f"  Fold {r['fold']}: {r['best_p_value_threshold']:.0e}")

        return pd.DataFrame(all_test_rows)


# --- Usage Example ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Emerson 2017 Disease Classification Evaluation")
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv file')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Root directory containing repertoire data files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Target disease to classify (e.g., Lupus, T1D, HIV)')
    parser.add_argument('--participant_col', default='participant_label')
    parser.add_argument('--disease_col', default='disease')
    parser.add_argument('--fold_col', default='malid_cross_validation_fold_id_when_in_test_set')
    parser.add_argument('--file_prefix', default='part_table_')
    parser.add_argument('--file_suffix', default='.tsv.gz')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV')
    parser.add_argument('--covariate_adjust', action='store_true',
                        help='Residualize per-sample diagnostic-TCR presence features against '
                             'demographics (age, sex, ancestry) and refit with L1 logistic regression. '
                             'Falls back to score-level adjustment if no diagnostic TCRs are found.')
    parser.add_argument('--adjust_distribution_by_demographics', action='store_true',
                        help='Apply per-disease cohort distribution adjustment for fair '
                             'comparison. HIV: filter both cohorts to African ancestry. '
                             'Lupus/T1D/Influenza/Covid19: keep the disease cohort unchanged '
                             'and subsample Healthy/Background so its age distribution (10y '
                             'bins) matches the disease cohort.')
    parser.add_argument('--random_baseline_seeds', type=int, nargs='+', default=None,
                        help='Run the random-sampling healthy baseline for each seed '
                             '(implies --adjust_distribution_by_demographics). For each '
                             'seed, healthy is resampled uniformly at random to the same '
                             'target N as the demographic-matched cohort; disease side '
                             'mirrors the demographic-matched run. Results from all seeds '
                             'are concatenated into one output, with a '
                             '`random_baseline_seed` column. Example: 7 14 21 28 35.')
    parser.add_argument('--ext_metadata_path', type=str, default=None,
                        help='Optional external-cohort metadata TSV (MAL-ID column style). '
                             'When set, external samples are merged into the same fold-based '
                             'CV split as the internal cohort and V/J genes are canonicalized.')
    parser.add_argument('--ext_data_dir', type=str, default=None,
                        help='Directory containing the external cohort repertoire files '
                             '(required when --ext_metadata_path is provided).')
    parser.add_argument('--ext_file_template', type=str,
                        default='{participant_label}_TCRB.tsv',
                        help='Filename template for external repertoires.')
    parser.add_argument('--max_folds', type=int, default=None,
                        help='Limit cross-validation to this many folds (default: all 3). '
                             'Useful for resource probes, e.g. --max_folds 1.')
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help='Directory for reusable fitted model artifacts.')
    args = parser.parse_args()

    if args.max_folds is not None and args.max_folds < 1:
        parser.error('--max_folds must be >= 1')

    print("Emerson 2017 Disease Classification Evaluation")
    print("=" * 60)
    print("\nTo run evaluation, ensure you have:")
    print("  1. Repertoire .tsv.gz files with a 'cdr3_aa' column (configurable)")
    print("  2. A metadata.tsv file with columns:")
    print("     - 'participant_label': e.g., 'BFI-0003052'")
    print("     - 'disease': e.g., 'Healthy/Background', 'Lupus', 'T1D'")
    print("     - 'malid_cross_validation_fold_id_when_in_test_set': fold ID (0, 1, 2)")
    print("\nFile names are constructed as: {prefix}{participant_label}_{specimen_label}{suffix}")
    print("  e.g., 'part_table_BFI-0003052_S001.tsv.gz'")
    print("\nBinary labels are created automatically based on the target disease.")
    
    # Initialize evaluator with custom train/val ratio (default is 0.9 for 9:1 split)
    evaluator = Emerson2017Evaluator(train_val_ratio=0.9, p_value_threshold=1e-4,
                                     sequence_col='cdr3_aa', v_col='v_call', j_col='j_call')
    metadata_path = args.metadata_path
    repertoire_data_dir = args.repertoire_data_dir
    RANDOM_SEED = 7

    # List available diseases
    diseases = evaluator.get_available_diseases(metadata_path)
    print(f"Available diseases: {diseases}")
    
    # Run 3-fold cross-validation for a specific disease WITH parameter tuning
    cv_kwargs = dict(
        metadata_path=metadata_path,
        target_disease=args.target_disease,
        data_dir=repertoire_data_dir,  # Root directory with data files
        participant_col=args.participant_col,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
        disease_col=args.disease_col,
        fold_col=args.fold_col,
        n_folds=args.max_folds if args.max_folds is not None else 3,
        model_save_dir=args.model_save_dir,
        random_state=RANDOM_SEED,
        tune_parameters=True,
        p_value_candidates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        covariate_adjust=args.covariate_adjust,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_file_template=args.ext_file_template,
    )

    if args.random_baseline_seeds:
        seed_dfs = []
        for seed in args.random_baseline_seeds:
            print(f"\n{'#'*60}")
            print(f"# RANDOM BASELINE RUN — seed={seed}")
            print(f"{'#'*60}")
            seed_df = evaluator.run_cross_validation(
                **cv_kwargs,
                adjust_distribution_by_demographics=True,
                random_baseline=True,
                random_baseline_seed=seed,
            )
            seed_dfs.append(seed_df)
        scores_df = pd.concat(seed_dfs, axis=0, ignore_index=True)

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
        print(f"\n{'#'*60}")
        print(f"# RANDOM BASELINE SUMMARY — across {len(summary_df)} seeds")
        print(f"{'#'*60}")
        print(summary_df.to_string(index=False))
        print(f"Mean overall AUROC: {summary_df['overall_auroc'].mean():.4f} "
              f"± {summary_df['overall_auroc'].std(ddof=0):.4f}")
        print(f"Mean overall AUPR:  {summary_df['overall_aupr'].mean():.4f} "
              f"± {summary_df['overall_aupr'].std(ddof=0):.4f}")
    else:
        scores_df = evaluator.run_cross_validation(
            **cv_kwargs,
            adjust_distribution_by_demographics=args.adjust_distribution_by_demographics,
        )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
    
    # Run 3-fold cross-validation WITHOUT parameter tuning
    # results = evaluator.run_cross_validation(
    #     metadata_path=metadata_path,
    #     target_disease='T1D',
    #     data_dir=repertoire_data_dir,
    #     n_folds=3,
    #     random_state=RANDOM_SEED,
    #     tune_parameters=False
    # )
