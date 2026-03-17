"""
Evaluation script for the ML Baseline (Gapped 4-mer + V/J gene) disease classification model.

Provides cross-validation functionality for evaluating Gapped_4mer_VJgene
on binary disease vs. Healthy/Background classification tasks.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.ml_baseline import Gapped_4mer_VJgene


class MLBaselineEvaluator:
    """
    Evaluator for the Gapped 4-mer + V/J gene ensemble model.

    All hyperparameter tuning (C values via k-fold CV, ensemble alpha via
    validation sweep) is handled internally by the model's train() method,
    so the evaluator passes all non-test data directly to train().
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7):
        """
        Args:
            val_split: Internal val fraction used by the model for alpha tuning.
            n_cv_folds: CV folds used by the model for C tuning.
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            subsample_fraction: Fraction of reads to sample per repertoire.
            subsample_seed: Random seed for reproducibility.
        """
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.model = None

    # ------------------------------------------------------------------
    # Metadata helpers (shared pattern across evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        """
        Filter metadata to target disease vs. Healthy/Background and add binary labels.

        Returns:
            DataFrame with a 'label' column (1 = disease, 0 = healthy).
        """
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

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
                              n_folds=3):
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

        Returns:
            Dict with fold-level results and overall AUROC / AUPR.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                        file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        results = {
            'target_disease': target_disease,
            'fold_results': [],
            'all_probs': [],
            'all_labels': [],
        }

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
            )

            train_result = self.model.train(train_files, train_labels)

            # Evaluate on test set
            test_probs = []
            for fp in tqdm(test_files, desc="Testing"):
                result = self.model.predict_diagnosis(fp)
                test_probs.append(result['probability_positive'])

            test_probs = np.array(test_probs)
            test_labels_arr = np.array(test_labels)

            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr = average_precision_score(test_labels_arr, test_probs)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}")

            fold_result = {
                'fold': test_fold,
                'n_train': len(train_data),
                'n_test': len(test_data),
                'best_c_kmer': train_result['best_c_kmer'],
                'best_c_vj': train_result['best_c_vj'],
                'best_alpha': train_result['best_alpha'],
                'val_auroc': train_result['val_auroc'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
                'test_probs': test_probs.tolist(),
                'test_labels': test_labels,
            }
            results['fold_results'].append(fold_result)
            results['all_probs'].extend(test_probs.tolist())
            results['all_labels'].extend(test_labels)

        # Overall metrics (all test predictions concatenated across folds)
        all_probs = np.array(results['all_probs'])
        all_labels = np.array(results['all_labels'])
        results['overall_auroc'] = roc_auc_score(all_labels, all_probs)
        results['overall_aupr'] = average_precision_score(all_labels, all_probs)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        fold_aurocs = [r['test_auroc'] for r in results['fold_results']]
        fold_auprs = [r['test_aupr'] for r in results['fold_results']]
        print(f"Mean Test AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:  {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
        print(f"Overall AUROC (combined): {results['overall_auroc']:.4f}")
        print(f"Overall AUPR  (combined): {results['overall_aupr']:.4f}")

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Baseline (Gapped 4-mer + V/J gene) Disease Classification"
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
    args = parser.parse_args()

    evaluator = MLBaselineEvaluator(
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
    )

    print(f"\nOverall AUROC: {results['overall_auroc']:.4f}")
    print(f"Overall AUPR:  {results['overall_aupr']:.4f}")
