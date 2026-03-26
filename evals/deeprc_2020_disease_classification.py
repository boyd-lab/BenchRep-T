"""
Evaluation script for DeepRC (2020) disease classification model.

Equivalent to ostmeyer_2019_disease_classification.py but uses DeepRC.
Reads AIRR-format .tsv.gz repertoire files directly (no HDF5 conversion).

Reference: Widrich et al. 2020, "Modern Hopfield Networks and Attention for
Immune Repertoire Classification"
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Allow imports from models/DeepRC (no __init__.py there)
_DEEPRC_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'DeepRC'
)
sys.path.insert(0, _DEEPRC_MODELS_DIR)

from architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from training import train, evaluate
from dataset_readers import make_dataloaders_from_airr, log_sequence_count_scaling
from task_definitions import TaskDefinition, BinaryTarget


class DeepRC2020Evaluator:
    """
    Evaluator for DeepRC (2020) on binary disease classification.

    Reads AIRR .tsv.gz files directly via AIRRRepertoireDataset —
    no HDF5 conversion required.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, n_updates=int(1e4), evaluate_at=int(1e3),
                 sequence_col='cdr3_aa', count_col='duplicate_count',
                 train_val_ratio=0.9, random_state=7,
                 n_worker_processes=4, batch_size=4,
                 sample_n_sequences=int(1e4),
                 kernel_size=9, n_kernels=32,
                 device=None, results_dir='results/deeprc'):
        """
        Args:
            n_updates: Number of gradient updates for training.
            evaluate_at: Evaluate / check early stopping every N updates.
            sequence_col: AIRR column with CDR3 amino acid sequences.
            count_col: AIRR column with sequence counts; uses 1 if absent.
            train_val_ratio: Fraction of non-test data used for training
                             (remainder is validation for early stopping).
            random_state: Seed for train/val split.
            n_worker_processes: DataLoader worker processes.
            batch_size: Repertoires per mini-batch during training.
            sample_n_sequences: Sequences randomly sampled per repertoire
                                 during training (None = use all).
            kernel_size: CNN kernel size for sequence embedding.
            n_kernels: Number of CNN kernels.
            device: torch.device string (default: cuda:0 if available, else cpu).
            results_dir: Base directory for DeepRC checkpoint/tensorboard files.
        """
        self.n_updates = n_updates
        self.evaluate_at = evaluate_at
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.train_val_ratio = train_val_ratio
        self.random_state = random_state
        self.n_worker_processes = n_worker_processes
        self.batch_size = batch_size
        self.sample_n_sequences = sample_n_sequences
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.results_dir = results_dir

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Metadata helpers (same pattern as other evaluators)
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)

        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}' classification:")
        print(f"  Disease ({target_disease}): {n_disease} samples")
        print(f"  Healthy ({self.HEALTHY_LABEL}): {n_healthy} samples")
        print(f"  Total: {len(filtered)} samples")
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
    # Model construction (mirrors cmv.py architecture)
    # ------------------------------------------------------------------

    def _build_model(self, task_definition):
        seq_emb = SequenceEmbeddingCNN(
            n_input_features=20 + 3,   # 20 AAs + 3 positional features
            kernel_size=self.kernel_size,
            n_kernels=self.n_kernels,
            n_layers=1,
        )
        attn_net = AttentionNetwork(
            n_input_features=self.n_kernels, n_layers=2, n_units=32)
        out_net = OutputNetwork(
            n_input_features=self.n_kernels,
            n_output_features=task_definition.get_n_output_features(),
            n_layers=1, n_units=32,
        )
        model = DeepRC(
            max_seq_len=30,
            sequence_embedding_network=seq_emb,
            attention_network=attn_net,
            output_network=out_net,
            consider_seq_counts=True,
            n_input_features=20,
            add_positional_information=True,
            sequence_reduction_fraction=0.1,
            reduction_mb_size=int(5e4),
            device=self.device,
        ).to(device=self.device)
        return model

    # ------------------------------------------------------------------
    # Per-sample inference (evaluate() only returns aggregate metrics)
    # ------------------------------------------------------------------

    def _predict_proba(self, model, dataloader):
        """Return (sample_ids, probabilities) for every sample in dataloader."""
        model.eval()
        all_ids = []
        all_probs = []
        with torch.no_grad():
            for targets, inputs, seq_lens, counts, sample_ids in dataloader:
                targets, inputs, seq_lens, n_seqs = model.reduce_and_stack_minibatch(
                    targets, inputs, seq_lens, counts)
                raw = model(inputs_flat=inputs, sequence_lengths_flat=seq_lens,
                            n_sequences_per_bag=n_seqs)
                probs = torch.sigmoid(raw[:, 0]).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_ids.extend(sample_ids)
        return all_ids, np.array(all_probs)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3,
                              allowed_participants=None):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        Non-test data is split into train / val; val is used for early stopping.
        AIRR files are read directly — no temporary files or HDF5 conversion.

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease label to classify against Healthy/Background.
            data_dir: Directory containing AIRR .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix: Filename prefix (default 'part_table_').
            file_suffix: Filename suffix (default '.tsv.gz').
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined test-fold IDs (0, 1, 2).
            n_folds: Number of cross-validation folds.
            allowed_participants: Optional set of specimen_labels to restrict to.

        Returns:
            pd.DataFrame with per-sample scores across all folds.
        """
        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                        file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants.")

        task_definition = TaskDefinition(
            targets=[BinaryTarget(column_name='label', true_class_value='1')]
        )

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
            train_val_data = metadata[~test_mask]

            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=self.random_state,
                stratify=train_val_data['label'],
            )
            print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            trainingset, trainingset_eval, validationset_eval, testset_eval = \
                make_dataloaders_from_airr(
                    task_definition=task_definition,
                    train_metadata=train_data,
                    val_metadata=val_data,
                    test_metadata=test_data,
                    file_path_col='file_path',
                    label_col='label',
                    sample_id_col='specimen_label',
                    sequence_col=self.sequence_col,
                    count_col=self.count_col,
                    sample_n_sequences=self.sample_n_sequences,
                    batch_size=self.batch_size,
                    n_worker_processes=self.n_worker_processes,
                    sequence_counts_scaling_fn=log_sequence_count_scaling,
                    verbose=True,
                )

            model = self._build_model(task_definition)
            fold_results_dir = os.path.join(self.results_dir,
                                             f'{target_disease}_fold{test_fold}')
            train(
                model=model,
                task_definition=task_definition,
                trainingset_dataloader=trainingset,
                trainingset_eval_dataloader=trainingset_eval,
                early_stopping_target_id='label',
                validationset_eval_dataloader=validationset_eval,
                n_updates=self.n_updates,
                evaluate_at=self.evaluate_at,
                device=self.device,
                results_directory=fold_results_dir,
            )

            test_ids, test_probs = self._predict_proba(model, testset_eval)
            test_labels_arr = np.array(test_data['label'].tolist())

            test_auroc = roc_auc_score(test_labels_arr, test_probs)
            test_aupr  = average_precision_score(test_labels_arr, test_probs)
            print(f"Test AUROC: {test_auroc:.4f}, Test AUPR: {test_aupr:.4f}")

            id_to_row = {row['specimen_label']: row
                         for _, row in test_data.iterrows()}
            for sid, score in zip(test_ids, test_probs):
                row = id_to_row[sid]
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': row['specimen_label'],
                    'disease_label': int(row['label']),
                    'disease_label_str': row[disease_col],
                    'method': 'DeepRC_2020',
                    'disease_model': target_disease,
                    'model_score': float(score),
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                })

            fold_results.append({
                'fold': test_fold,
                'test_auroc': test_auroc,
                'test_aupr':  test_aupr,
            })
            all_probs.extend(test_probs.tolist())
            all_labels.extend(test_labels_arr.tolist())

        all_probs_arr  = np.array(all_probs)
        all_labels_arr = np.array(all_labels)
        overall_auroc  = roc_auc_score(all_labels_arr, all_probs_arr)
        overall_aupr   = average_precision_score(all_labels_arr, all_probs_arr)

        print(f"\n{'='*60}")
        print(f"OVERALL RESULTS: {target_disease} vs Healthy")
        print(f"{'='*60}")
        fold_aurocs = [r['test_auroc'] for r in fold_results]
        fold_auprs  = [r['test_aupr']  for r in fold_results]
        print(f"Mean Test AUROC: {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
        print(f"Mean Test AUPR:  {np.mean(fold_auprs):.4f}  ± {np.std(fold_auprs):.4f}")
        print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
        print(f"Overall AUPR  (all folds combined): {overall_aupr:.4f}")

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepRC 2020 Disease Classification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing AIRR .tsv.gz repertoire files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--n_updates', type=int, default=int(1e4),
                        help='Number of gradient updates (default: 10000)')
    parser.add_argument('--evaluate_at', type=int, default=int(1e3),
                        help='Evaluate every N updates (default: 1000)')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device string, e.g. "cuda:0" or "cpu"')
    parser.add_argument('--results_dir', type=str, default='results/deeprc',
                        help='Directory for DeepRC checkpoints/tensorboard files')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Repertoires per mini-batch (default: 4)')
    parser.add_argument('--sample_n_sequences', type=int, default=int(1e4),
                        help='Sequences sampled per repertoire during training (default: 10000)')
    args = parser.parse_args()

    evaluator = DeepRC2020Evaluator(
        n_updates=args.n_updates,
        evaluate_at=args.evaluate_at,
        device=args.device,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        sample_n_sequences=args.sample_n_sequences,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
