"""
Evaluation script for the ABMIL (Gapped 4-mer + V/J gene) disease classification model.

Applies Gated Attention Multiple Instance Learning over per-sequence gapped 4-mer
and V/J gene features. Each repertoire is treated as a bag of sequence instances;
the ABMIL model learns to weight informative sequences via attention pooling.

Reference architecture: Ilse et al. 2018, "Attention-based Deep Multiple Instance
Learning" (https://arxiv.org/abs/1802.04712).
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from airr_bench.models.ensemble_abmil import ABMIL_4mer_VJgene


class ABMILEvaluator:
    """
    Evaluator for the ABMIL gapped 4-mer + V/J gene model.

    All feature extraction and hyperparameter choices are handled internally
    by ABMIL_4mer_VJgene; the evaluator passes all non-test data directly to
    model.train() for each cross-validation fold.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(
        self,
        sequence_col='cdr3_aa',
        v_gene_col='v_call',
        j_gene_col='j_call',
        max_instances=500,
        M=256,
        L=128,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-5,
        patience=10,
        val_split=0.2,
        seed=7,
        subsample_fraction=1.0,
        subsample_seed=7,
        use_gpu=True,
        indices_map=None,
    ):
        """
        Args:
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls.
            max_instances: Maximum sequences sampled per bag (memory / regularisation).
            M: ABMIL encoder / bag-embedding hidden dimension.
            L: ABMIL attention hidden dimension.
            epochs: Maximum training epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Early-stopping patience in epochs.
            val_split: Fraction of training bags held out for early stopping.
            seed: Random seed for val split and subsampling.
            subsample_fraction: Fraction of reads to sample per repertoire.
            subsample_seed: Random seed for repertoire subsampling.
            use_gpu: Use CUDA if available.
            indices_map: Dict mapping specimen_label to pre-computed row indices.
        """
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
        self.indices_map = indices_map
        self.model = None

    # ------------------------------------------------------------------
    # Metadata helpers (shared pattern across evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease',
                             require_demographics=False):
        """
        Filter metadata to target disease vs. Healthy/Background and add binary labels.

        Args:
            metadata: DataFrame with metadata.
            target_disease: Disease name to classify.
            disease_col: Column with disease labels.
            require_demographics: If True, drop rows with missing age, sex,
                or ancestry so the subset matches the demographic baseline.

        Returns:
            DataFrame with a 'label' column (1 = disease, 0 = healthy).
        """
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

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
                              allowed_participants=None,
                              require_demographics=False):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold, all non-test samples are passed to model.train(), which
        fits the DictVectorizer, scaler, and ABMIL network from scratch.

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
            allowed_participants: Optional set of specimen labels to restrict to.
            require_demographics: If True, drop repertoires with missing
                                  demographic data (age, sex, ancestry).

        Returns:
            DataFrame with per-sample scores across all folds.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col,
                                             require_demographics=require_demographics)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

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
            self.model = ABMIL_4mer_VJgene(
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
                use_gpu=self.use_gpu,
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

            for (_, row), score in zip(test_data.iterrows(), test_probs):
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'method': 'ABMIL_Ensemble',
                    'disease_model': target_disease,
                    'model_score': float(score),
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                })

            fold_results.append({
                'fold': test_fold,
                'best_val_loss': train_result['best_val_loss'],
                'epochs_trained': train_result['epochs_trained'],
                'test_auroc': test_auroc,
                'test_aupr': test_aupr,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels)

        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        fold_aurocs = [r['test_auroc'] for r in fold_results]
        fold_auprs = [r['test_aupr'] for r in fold_results]
        print(f"Mean Test AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:  {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")
        print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
        print(f"Overall AUPR  (all folds combined): {overall_aupr:.4f}")

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABMIL (Gapped 4-mer + V/J gene) Disease Classification"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--max_instances', type=int, default=500,
                        help='Max sequences sampled per bag (default: 500)')
    parser.add_argument('--M', type=int, default=256,
                        help='ABMIL encoder hidden dim (default: 256)')
    parser.add_argument('--L', type=int, default=128,
                        help='ABMIL attention hidden dim (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Adam learning rate (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early-stopping patience in epochs (default: 10)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of train bags held out for early stopping (default: 0.2)')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU even if CUDA is available')
    parser.add_argument('--require_demographics', action='store_true',
                        help='Drop repertoires with missing demographic data '
                             '(age, sex, ancestry) to match demographic baseline subset')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    args = parser.parse_args()

    evaluator = ABMILEvaluator(
        max_instances=args.max_instances,
        M=args.M,
        L=args.L,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        val_split=args.val_split,
        use_gpu=not args.no_gpu,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        require_demographics=args.require_demographics,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
