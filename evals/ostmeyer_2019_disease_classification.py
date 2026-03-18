"""
Evaluation script for Ostmeyer 2019 MIL-TCR classification model.

This module provides cross-validation and parameter tuning functionality
for evaluating the MIL_TCR_Classifier on disease classification tasks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.ostmeyer_2019 import MIL_TCR_Classifier


class Ostmeyer2019Evaluator:
    """
    Evaluator for the Ostmeyer 2019 MIL-TCR Classifier.
    
    Provides functionality for:
    - Loading metadata with fold assignments
    - Creating binary labels for disease vs healthy classification
    - Hyperparameter tuning using validation sets
    - K-fold cross-validation with pre-defined folds
    """
    
    # Constants for label values
    HEALTHY_LABEL = "Healthy/Background"
    
    def __init__(self, train_val_ratio=0.9, n_restarts=250_000, max_iter=2500,
                 learning_rate=0.1, abundance_method='A', sequence_col='cdr3_aa',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None):
        """
        Initialize the evaluator.

        Args:
            train_val_ratio: Ratio of training data in train/val split (default: 0.9, i.e., 9:1)
            n_restarts: Number of random restarts for optimization (default: 250,000,
                        matching the paper's best models). Reduce for faster runs.
            max_iter: Number of gradient descent steps per restart (default: 2500,
                      matching the paper exactly).
            learning_rate: Step size for gradient descent (default: 0.1).
            abundance_method: 'A' for 4-mer relative abundance or 'B' for TCRb relative
                              abundance (default: 'A')
            sequence_col: Column name containing TCR sequences in repertoire files (default: 'cdr3_aa')
            subsample_fraction: Fraction of reads to keep for depth simulation (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling (default: 7)
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set)
        """
        self.train_val_ratio = train_val_ratio
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.abundance_method = abundance_method
        self.sequence_col = sequence_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
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
    
    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        """
        Prepare binary classification data for a specific disease.
        
        Filters the metadata to include only:
        - Samples with the target disease (label = 1)
        - Healthy/Background samples (label = 0)
        
        Args:
            metadata: DataFrame with metadata
            target_disease: Name of the disease to classify (e.g., 'Lupus', 'T1D', 'HIV')
            disease_col: Column name containing disease labels (default: 'disease')
        
        Returns:
            DataFrame with filtered data and a new 'label' column (1 for disease, 0 for healthy)
        """
        # Filter to include only target disease and healthy samples
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered_data = metadata[mask].copy()
        
        # Create binary labels: 1 for disease, 0 for healthy
        filtered_data['label'] = (filtered_data[disease_col] == target_disease).astype(int)
        
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
                       abundance_method_candidates=None):
        """
        Optionally tune abundance_method using validation set, then train.

        The paper does not tune hyperparameters in the traditional sense — it
        selects the best model across random restarts by training loss. The only
        meaningful choice left is abundance_method ('A' vs 'B'). If
        abundance_method_candidates is provided, both are evaluated on the
        validation set and the better one is used.

        Args:
            train_files: List of training file paths
            train_labels: List of training labels
            val_files: List of validation file paths
            val_labels: List of validation labels
            abundance_method_candidates: List of abundance methods to try,
                                         e.g. ['A', 'B']. If None, uses
                                         self.abundance_method directly without
                                         tuning.

        Returns:
            Dictionary with tuning results and best parameters
        """
        if abundance_method_candidates is None:
            abundance_method_candidates = [self.abundance_method]

        print("--- Parameter Tuning ---")
        print(f"Testing abundance methods: {abundance_method_candidates}")

        all_files = list(train_files) + list(val_files)

        tuning_results = []

        print("\n--- Evaluating abundance methods ---")
        for method in abundance_method_candidates:
            model = MIL_TCR_Classifier(
                n_restarts=self.n_restarts,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                abundance_method=method,
                sequence_col=self.sequence_col,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed
            )

            # Preload and pre-extract features once per method
            model.preload_repertoires(all_files)
            for file_path in tqdm(all_files, desc=f"Extracting features (method={method})"):
                model.extract_4mer_features(file_path, use_cache=True)

            try:
                train_result = model.train(train_files, train_labels)
            except Exception as e:
                print(f"  method={method}: Training failed: {e}")
                tuning_results.append({
                    'abundance_method': method,
                    'val_auroc': 0.0,
                    'val_aupr': 0.0,
                    'best_loss': float('inf')
                })
                continue

            val_probs = []
            for file_path in val_files:
                result = model.predict_diagnosis(file_path)
                val_probs.append(result['probability_positive'])

            val_probs = np.array(val_probs)
            val_labels_arr = np.array(val_labels)

            try:
                val_auroc = roc_auc_score(val_labels_arr, val_probs)
                val_aupr = average_precision_score(val_labels_arr, val_probs)
            except Exception:
                val_auroc = 0.5
                val_aupr = 0.5

            tuning_results.append({
                'abundance_method': method,
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
                'best_loss': train_result['best_loss']
            })

            print(f"  method={method}: Val AUROC={val_auroc:.4f}, Val AUPR={val_aupr:.4f}")

        best_result = max(tuning_results, key=lambda x: x['val_auroc'])
        best_method = best_result['abundance_method']

        print(f"\nBest abundance method: {best_method} "
              f"(Val AUROC={best_result['val_auroc']:.4f}, Val AUPR={best_result['val_aupr']:.4f})")

        # Final model with best method
        self.model = MIL_TCR_Classifier(
            n_restarts=self.n_restarts,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            abundance_method=best_method,
            sequence_col=self.sequence_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n
        )
        self.model.preload_repertoires(all_files)
        for file_path in tqdm(all_files, desc="Extracting features (final model)"):
            self.model.extract_4mer_features(file_path, use_cache=True)
        self.model.train(train_files, train_labels)

        return {
            'tuning_results': tuning_results,
            'best_abundance_method': best_method,
            'best_val_auroc': best_result['val_auroc'],
            'best_val_aupr': best_result['val_aupr']
        }
    
    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3, random_state=7,
                              tune_parameters=True,
                              abundance_method_candidates=None,
                              allowed_participants=None):
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
            tune_parameters: Whether to tune abundance_method using validation set
                             (default: True)
            abundance_method_candidates: List of abundance methods to try during tuning,
                                         e.g. ['A', 'B']. If None, uses ['A', 'B'].

        Returns:
            Dictionary containing results for each fold and overall metrics
        """
        # Load and prepare data with binary labels
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        
        # Add file paths
        metadata = self.add_file_paths(
            metadata, data_dir, participant_col, file_prefix, file_suffix
        )
        
        # Filter to only include files that exist
        metadata = self.filter_existing_files(metadata)

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

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            # Split data by fold
            test_mask = metadata[fold_col] == test_fold
            train_val_mask = ~test_mask

            test_data = metadata[test_mask]
            train_val_data = metadata[train_val_mask]

            # Further split train_val into train and validation
            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=random_state,
                stratify=train_val_data['label']  # Use binary label column
            )

            print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

            # Extract file paths and labels
            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()

            val_files = val_data['file_path'].tolist()
            val_labels = val_data['label'].tolist()

            test_files = test_data['file_path'].tolist()
            test_labels = test_data['label'].tolist()

            # Train with or without parameter tuning
            if tune_parameters:
                # Use validation set to tune abundance_method
                tuning_result = self.tune_and_train(
                    train_files, train_labels,
                    val_files, val_labels,
                    abundance_method_candidates=abundance_method_candidates
                )
                best_method = tuning_result['best_abundance_method']
                val_auroc = tuning_result['best_val_auroc']
                val_aupr = tuning_result['best_val_aupr']
            else:
                # Train without tuning (use self.abundance_method directly)
                self.model = MIL_TCR_Classifier(
                    n_restarts=self.n_restarts,
                    max_iter=self.max_iter,
                    learning_rate=self.learning_rate,
                    abundance_method=self.abundance_method,
                    sequence_col=self.sequence_col,
                    subsample_fraction=self.subsample_fraction,
                    subsample_seed=self.subsample_seed
                )
                self.model.train(train_files, train_labels)

                # Evaluate on validation set
                val_probs = []
                for file_path in tqdm(val_files, desc="Validating", leave=False):
                    result = self.model.predict_diagnosis(file_path)
                    val_probs.append(result['probability_positive'])
                val_probs = np.array(val_probs)
                val_labels_arr = np.array(val_labels)
                val_auroc = roc_auc_score(val_labels_arr, val_probs)
                val_aupr = average_precision_score(val_labels_arr, val_probs)
                best_method = self.abundance_method
                tuning_result = None

            print(f"\nFinal Validation AUROC: {val_auroc:.4f}, AUPR: {val_aupr:.4f}")

            # Evaluate on test set
            test_probs = []
            for file_path in tqdm(test_files, desc="Testing"):
                result = self.model.predict_diagnosis(file_path)
                test_probs.append(result['probability_positive'])

            test_probs = np.array(test_probs)
            test_labels_arr = np.array(test_labels)

            # Compute AUROC and AUPR for test set
            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}")

            # Build per-sample rows for output DataFrame
            for (_, row), score in zip(test_data.iterrows(), test_probs):
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'method': 'Ostmeyer_2019',
                    'disease_model': target_disease,
                    'model_score': float(score),
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                })

            fold_results.append({
                'fold': test_fold,
                'best_abundance_method': best_method,
                'val_auroc': val_auroc,
                'val_aupr': val_aupr,
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
            })

            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels)

            # Clear cache between folds to save memory
            self.model.clear_cache()

        # Calculate overall metrics across all folds
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)

        print(f"\n{'='*60}")
        print(f"OVERALL CROSS-VALIDATION RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        print(f"Mean Test AUROC: {np.mean([r['test_auroc'] for r in fold_results]):.4f} "
              f"± {np.std([r['test_auroc'] for r in fold_results]):.4f}")
        print(f"Mean Test AUPR:  {np.mean([r['test_aupr'] for r in fold_results]):.4f} "
              f"± {np.std([r['test_aupr'] for r in fold_results]):.4f}")
        print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
        print(f"Overall AUPR (all folds combined):  {overall_aupr:.4f}")

        if tune_parameters:
            print(f"\nBest abundance method per fold:")
            for r in fold_results:
                print(f"  Fold {r['fold']}: abundance_method={r['best_abundance_method']}")

        return pd.DataFrame(all_test_rows)


# --- Usage Example ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ostmeyer 2019 Disease Classification Evaluation")
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv file')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Root directory containing repertoire data files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Target disease to classify (e.g., Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV')
    args = parser.parse_args()
    
    print("Ostmeyer 2019 Disease Classification Evaluation")
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
    evaluator = Ostmeyer2019Evaluator(
        train_val_ratio=0.9,
        n_restarts=250_000,
        max_iter=2500,
        learning_rate=0.1,
        abundance_method='A',
        sequence_col='cdr3_aa'
    )
    metadata_path = args.metadata_path
    repertoire_data_dir = args.repertoire_data_dir
    RANDOM_SEED = 7

    # List available diseases
    diseases = evaluator.get_available_diseases(metadata_path)
    print(f"Available diseases: {diseases}")
    
    # Run 3-fold cross-validation for a specific disease WITH parameter tuning
    scores_df = evaluator.run_cross_validation(
        metadata_path=metadata_path,
        target_disease=args.target_disease,
        data_dir=repertoire_data_dir,  # Root directory with data files
        participant_col='participant_label',
        file_prefix='part_table_',
        file_suffix='.tsv.gz',
        disease_col='disease',
        fold_col='malid_cross_validation_fold_id_when_in_test_set',
        n_folds=3,
        random_state=RANDOM_SEED,
        tune_parameters=True,
        abundance_method_candidates=['A', 'B']
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
