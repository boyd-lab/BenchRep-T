"""
Driver sequence identification evaluation for DeepRC (2020).

Per-sequence importance is the softmax attention weight assigned by the trained
AttentionNetwork to each CDR3 sequence.  The attention network processes CNN
sequence embeddings (h()) and outputs a scalar importance score; after softmax
normalization over all sequences in a repertoire, higher values indicate the
model focused more on that sequence when forming its prediction.

To extract attention weights for ALL sequences (not just the top-10% normally
retained by `reduce_and_stack_minibatch`), the DeepRC model's internal
`__compute_features__` and `sequence_embedding` + `attention_nn` modules are
called directly on the full repertoire.

Multiple rows sharing the same CDR3 receive the max attention weight across
those rows.

Sequences are ranked descending by attention weight and the top-k CDR3s are
compared against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_DEEPRC_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'DeepRC'
)
sys.path.insert(0, _DEEPRC_MODELS_DIR)

from architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from training import train
from dataset_readers import (make_dataloaders_from_airr,
                              no_sequence_count_scaling,
                              AIRRRepertoireDataset)
from task_definitions import TaskDefinition, BinaryTarget


class DeepRC2020DriverIdentificationEvaluator:
    """
    Evaluates DeepRC (2020) driver sequence identification using per-sequence
    attention weights from the trained AttentionNetwork.
    """

    HEALTHY_LABEL = "Healthy/Background"

    # AA vocabulary (same as AIRRRepertoireDataset._AAS)
    _AAS = 'ACDEFGHIKLMNPQRSTVWY'

    @staticmethod
    def _find_checkpoint(model_save_dir, target_disease, test_fold):
        """Return path to best_u*.tar.gzip in the fold's checkpoint dir, or None."""
        import glob
        pattern = os.path.join(model_save_dir,
                               f'{target_disease}_fold{test_fold}',
                               '*', 'checkpoint', 'best_u*.tar.gzip')
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def _load_model_from_checkpoint(self, checkpoint_path, task_definition):
        """Load a DeepRC model from a SaverLoader checkpoint without retraining."""
        import tempfile
        from widis_lstm_tools.utils.collection import SaverLoader
        model = self._build_model(task_definition)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        state = dict(model=model, optimizer=optimizer, update=0,
                     best_validation_loss=np.inf)
        tmp_dir = tempfile.mkdtemp()
        saver_loader = SaverLoader(save_dict=state, device=self.device,
                                   save_dir=tmp_dir, n_savefiles=1, n_inmem=1)
        state.update(saver_loader.load_from_file(loadname=checkpoint_path, verbose=True))
        model.to(self.device)
        model.eval()
        return model

    def __init__(self, n_updates=int(1e4), evaluate_at=int(1e3),
                 sequence_col='cdr3_aa', count_col='duplicate_count',
                 train_val_ratio=0.9, random_state=7,
                 n_worker_processes=4, batch_size=32,
                 sample_n_sequences=int(1e4),
                 kernel_size=9, n_kernels=32, max_seq_len=50,
                 device=None, results_dir='results/deeprc',
                 indices_map=None):
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
        self.max_seq_len = max_seq_len
        self.results_dir = results_dir
        self.indices_map = indices_map

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Build byte-level AA lookup (same as AIRRRepertoireDataset)
        self._aa_to_idx = {c: i for i, c in enumerate(self._AAS)}
        self._byte_lookup = np.full(256, -1, dtype=np.int8)
        for char, idx in self._aa_to_idx.items():
            self._byte_lookup[ord(char)] = idx

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path):
        return pd.read_csv(metadata_path, sep='\t')

    def prepare_disease_data(self, metadata, target_disease, disease_col='disease'):
        mask = metadata[disease_col].isin([target_disease, self.HEALTHY_LABEL])
        filtered = metadata[mask].copy()
        filtered['label'] = (filtered[disease_col] == target_disease).astype(int)
        n_disease = (filtered['label'] == 1).sum()
        n_healthy = (filtered['label'] == 0).sum()
        print(f"Prepared data for '{target_disease}': "
              f"{n_disease} disease, {n_healthy} healthy, {len(filtered)} total")
        return filtered

    def add_file_paths(self, metadata, data_dir, participant_col='participant_label',
                       file_prefix='part_table_', file_suffix='.tsv.gz'):
        metadata = metadata.copy()
        metadata['file_path'] = metadata.apply(
            lambda row: os.path.join(
                data_dir,
                f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}"
            ), axis=1
        )
        return metadata

    def filter_existing_files(self, metadata):
        original = len(metadata)
        metadata = metadata.copy()
        metadata['file_exists'] = metadata['file_path'].apply(os.path.exists)
        filtered = metadata[metadata['file_exists']].drop(columns=['file_exists'])
        missing = original - len(filtered)
        if missing > 0:
            print(f"Note: {missing}/{original} files not found. "
                  f"Proceeding with {len(filtered)}.")
        return filtered

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def load_driver_sequences(self, driver_seqs_path, target_disease):
        df = pd.read_csv(driver_seqs_path)
        disease_df = df[df['disease'] == target_disease]
        drivers_by_file = {}
        for filename, group in disease_df.groupby('filename'):
            drivers_by_file[filename] = set(group['sample_cdr3'].unique())
        total = sum(len(v) for v in drivers_by_file.values())
        print(f"Ground truth for '{target_disease}': "
              f"{len(drivers_by_file)} repertoires, {total} driver CDR3s")
        return drivers_by_file

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_model(self, task_definition):
        seq_emb = SequenceEmbeddingCNN(
            n_input_features=20 + 3,
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
            max_seq_len=self.max_seq_len,
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
    # Per-sequence attention extraction
    # ------------------------------------------------------------------

    def _encode_repertoire(self, file_path):
        """
        Load and encode a single AIRR repertoire into DeepRC's feature format.

        Returns:
            features: float tensor (K, max_seq_len, n_features) on device
            seq_lengths: long tensor (K,) on device
            counts: float tensor (K,) on device (raw, matching no_sequence_count_scaling)
            cdr3_strings: list of K CDR3 strings (original order)
        """
        try:
            df = pd.read_csv(file_path, sep='\t',
                             usecols=lambda c: c in {self.sequence_col, self.count_col})
        except Exception as e:
            print(f"Warning: could not read {file_path}: {e}")
            return None, None, None, []

        if self.sequence_col not in df.columns or len(df) == 0:
            return None, None, None, []

        if self.count_col not in df.columns:
            df[self.count_col] = 1

        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
            indices = self.indices_map.get(rep_id)
            if indices is not None:
                df = df.iloc[indices]

        raw_seqs = df[self.sequence_col].astype(str).values
        raw_counts = pd.to_numeric(df[self.count_col], errors='coerce').fillna(1).values.astype(np.float32)

        # Filter to sequences composed entirely of valid AAs, matching training path
        def _all_valid(s):
            if not s:
                return False
            b = np.frombuffer(s.encode('ascii', errors='replace'), dtype=np.uint8)
            return bool(np.all(self._byte_lookup[b] >= 0))

        valid_mask = np.array([_all_valid(s) for s in raw_seqs], dtype=bool)
        seqs = raw_seqs[valid_mask].tolist()
        raw_counts = raw_counts[valid_mask]

        if len(seqs) == 0:
            return None, None, None, []

        # Use raw counts, matching no_sequence_count_scaling used during training
        raw_counts = np.maximum(raw_counts, 0)

        K = len(seqs)
        max_len = self.max_seq_len
        n_features = len(self._AAS) + 3  # 20 AAs + 3 positional

        # Encode as int8 index arrays (-1 = unknown/pad)
        seq_indices = np.full((K, max_len), -1, dtype=np.int8)
        seq_lengths = np.zeros(K, dtype=np.int64)
        for i, seq in enumerate(seqs):
            arr = np.frombuffer(seq[:max_len].encode(), dtype=np.uint8)
            idx = self._byte_lookup[arr]
            L = len(idx)
            seq_indices[i, :L] = idx
            seq_lengths[i] = L

        seq_idx_t = torch.from_numpy(seq_indices).long().to(self.device)
        seq_len_t = torch.from_numpy(seq_lengths).long().to(self.device)
        counts_t = torch.from_numpy(raw_counts.astype(np.float32)).to(self.device)

        # Replicate __compute_features__ from DeepRC (architectures.py)
        embedding_dtype = torch.float16
        features = torch.zeros(
            (K, max_len, n_features), dtype=embedding_dtype, device=self.device
        )
        # One-hot encoding
        flat_idx = seq_idx_t.reshape(-1)
        flat_features = features[:, :seq_idx_t.shape[1], :].reshape(-1, n_features)
        valid_mask = flat_idx != -1
        flat_features[torch.arange(flat_features.shape[0], device=self.device)[valid_mask],
                      flat_idx[valid_mask]] = 1.0
        features[:, :seq_idx_t.shape[1], :] = flat_features.reshape(K, seq_idx_t.shape[1], n_features)
        # Scale by counts
        features = features * counts_t[:, None, None]
        # Add positional information
        pos_feats = torch.from_numpy(
            _compute_position_features_np(max_len, seq_lengths)
        ).to(dtype=embedding_dtype, device=self.device)
        features[:, :seq_idx_t.shape[1], -3:] = pos_feats[:, :seq_idx_t.shape[1]]
        # Normalize
        std = features.std()
        if std > 0:
            features = features / std

        return features, seq_len_t, counts_t, seqs

    def score_repertoire_cdr3s(self, file_path, model):
        """
        Score each CDR3 in a repertoire using the trained attention weights.

        All sequences are processed through CNN (h()) → attention network (f())
        → softmax.  Higher softmax weight = model relied more on that sequence.

        Returns:
            List of (cdr3, attention_weight) sorted descending.
        """
        model.eval()

        features, seq_lengths, _counts, cdr3_strings = self._encode_repertoire(file_path)
        if features is None or len(cdr3_strings) == 0:
            return []

        K = features.shape[0]

        with torch.no_grad():
            # Run CNN in minibatches to avoid OOM on large repertoires
            mb_size = int(5e4)
            emb_parts = []
            for start in range(0, K, mb_size):
                chunk = features[start:start + mb_size].to(dtype=torch.float16)
                lens_chunk = seq_lengths[start:start + mb_size]
                emb_parts.append(model.sequence_embedding(chunk, sequence_lengths=lens_chunk))
            emb_seqs = torch.cat(emb_parts, dim=0)  # (K, n_kernels)

            # Attention weights (pre-softmax)
            attn_raw = model.attention_nn(emb_seqs.to(dtype=torch.float32))  # (K, 1)
            attn = torch.softmax(attn_raw, dim=0).squeeze(dim=1).cpu().numpy()  # (K,)

        # Aggregate by CDR3 — max attention weight per unique CDR3
        cdr3_scores = {}
        for cdr3, weight in zip(cdr3_strings, attn):
            cdr3_scores[cdr3] = max(cdr3_scores.get(cdr3, 0.0), float(weight))

        return sorted(cdr3_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Main cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                             driver_seqs_path, ks,
                             participant_col='participant_label',
                             file_prefix='part_table_', file_suffix='.tsv.gz',
                             disease_col='disease',
                             fold_col='malid_cross_validation_fold_id_when_in_test_set',
                             n_folds=3, random_state=None,
                             allowed_participants=None,
                             max_repertoires=None,
                             model_save_dir=None,
                             raw_file_cache=None,
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification.

        For each fold:
        1. Train DeepRC on non-test data (val split for early stopping).
        2. Extract per-sequence attention weights for each test repertoire.
        3. Compare top-k CDR3s against ground truth driver sequences for each k in ks.

        Returns:
            DataFrame with per-repertoire, per-k precision@k and recall@k.
        """
        rng = random_state if random_state is not None else self.random_state

        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)}/{before} specimens "
                  f"via allowed_participants.")

        if max_repertoires is not None and len(metadata) > max_repertoires:
            metadata = metadata.sample(n=max_repertoires, random_state=self.random_state)
            print(f"Subsampled to {max_repertoires} repertoires for debug run.")

        task_definition = TaskDefinition(
            targets=[BinaryTarget(column_name='label', true_class_value='1')]
        )

        drivers_by_file = self.load_driver_sequences(driver_seqs_path, target_disease)

        all_results = []
        fold_summaries = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            test_mask = metadata[fold_col] == test_fold
            test_data = metadata[test_mask]
            train_val_data = metadata[~test_mask]

            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=rng,
                stratify=train_val_data['label'],
            )
            print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            trainingset, trainingset_eval, validationset_eval, _ = \
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
                    sequence_counts_scaling_fn=no_sequence_count_scaling,
                    indices_map=self.indices_map,
                    raw_file_cache=raw_file_cache,
                    verbose=True,
                )

            checkpoint_path = (self._find_checkpoint(model_save_dir, target_disease, test_fold)
                               if model_save_dir else None)
            if checkpoint_path is not None:
                print(f"  Loading model from checkpoint: {checkpoint_path}")
                model = self._load_model_from_checkpoint(checkpoint_path, task_definition)
            else:
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

            print(f"\n--- Scoring test repertoires ---")
            fold_precisions = {k: [] for k in ks}
            fold_recalls = {k: [] for k in ks}

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Scoring"):
                file_path = row['file_path']
                filename_stem = os.path.basename(file_path) \
                    .replace('.tsv.gz', '').replace('.tsv', '')

                if filename_stem not in drivers_by_file:
                    continue

                driver_cdr3s = drivers_by_file[filename_stem]
                ranked = self.score_repertoire_cdr3s(file_path, model)

                for k in ks:
                    top_k_cdr3s = set(cdr3 for cdr3, _ in ranked[:k])
                    hits = top_k_cdr3s & driver_cdr3s
                    precision = len(hits) / k
                    recall = len(hits) / len(driver_cdr3s)

                    fold_precisions[k].append(precision)
                    fold_recalls[k].append(recall)

                    all_results.append({
                        'fold': test_fold,
                        'k': k,
                        'filename': filename_stem,
                        'participant_label': row[participant_col],
                        'disease_label': int(row['label']),
                        'n_repertoire_unique_cdr3s': len(ranked),
                        'n_ground_truth_drivers': len(driver_cdr3s),
                        'n_hits_at_k': len(hits),
                        'precision_at_k': precision,
                        'recall_at_k': recall,
                    })

            n_reps = len(fold_precisions[ks[0]])
            if n_reps > 0:
                print(f"\nFold {test_fold}: {n_reps} repertoires")
                for k in ks:
                    mean_prec = np.mean(fold_precisions[k])
                    mean_rec = np.mean(fold_recalls[k])
                    print(f"  Mean Precision@{k}: {mean_prec:.4f}")
                    print(f"  Mean Recall@{k}:    {mean_rec:.4f}")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': n_reps,
                    **{f'mean_precision_at_{k}': np.mean(fold_precisions[k]) for k in ks},
                    **{f'mean_recall_at_{k}': np.mean(fold_recalls[k]) for k in ks},
                })
            else:
                print(f"\nFold {test_fold}: No test repertoires with ground truth drivers")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': 0,
                    **{f'mean_precision_at_{k}': float('nan') for k in ks},
                    **{f'mean_recall_at_{k}': float('nan') for k in ks},
                })

            del model

        results_df = pd.DataFrame(all_results)

        print(f"\n{'=' * 60}")
        print(f"OVERALL RESULTS: {target_disease} Driver Identification")
        print(f"{'=' * 60}")

        for k in ks:
            k_df = results_df[results_df['k'] == k] if len(results_df) > 0 else results_df
            if len(k_df) > 0:
                overall_precision = k_df['precision_at_k'].mean()
                overall_recall = k_df['recall_at_k'].mean()
                total_hits = k_df['n_hits_at_k'].sum()
                total_possible = k_df['n_ground_truth_drivers'].sum()
                print(f"\nk={k} ({len(k_df)} repertoires):")
                print(f"  Overall Precision@{k} (macro): {overall_precision:.4f}")
                print(f"  Overall Recall@{k}    (macro): {overall_recall:.4f}")
                if total_possible > 0:
                    micro_recall = total_hits / total_possible
                    micro_precision = total_hits / (len(k_df) * k)
                    print(f"  Overall Precision@{k} (micro): {micro_precision:.4f}")
                    print(f"  Overall Recall@{k}    (micro): {micro_recall:.4f}")
            else:
                print(f"\nk={k}: No results")

        print(f"\nPer-fold breakdown:")
        for s in fold_summaries:
            k_strs = '  '.join(
                f"P@{k}={s[f'mean_precision_at_{k}']:.4f} R@{k}={s[f'mean_recall_at_{k}']:.4f}"
                for k in ks
            )
            print(f"  Fold {s['fold']}: {s['n_repertoires']} reps | {k_strs}")

        if output_csv and len(results_df) > 0:
            results_df.to_csv(output_csv, index=False)
            print(f"\nPer-repertoire results saved to: {output_csv}")

        return results_df


# ---------------------------------------------------------------------------
# Position feature helper (replicates compute_position_features from architectures.py)
# ---------------------------------------------------------------------------

def _compute_position_features_np(max_seq_len, sequence_lengths, dtype=np.float16):
    """Compute 3-channel positional features for each sequence in a batch."""
    sequences = np.zeros((len(sequence_lengths), max_seq_len, 3), dtype=dtype)
    half_seq_lens = np.ceil(sequence_lengths / 2.).astype(int)
    for i, (seq_len, half_seq_len) in enumerate(zip(sequence_lengths, half_seq_lens)):
        if seq_len == 0:
            continue
        ramp = np.abs(0.5 - np.linspace(1.0, 0, num=seq_len)) * 2.
        sequences[i, :half_seq_len, 0] = ramp[:half_seq_len]
        sequences[i, half_seq_len:seq_len, 1] = ramp[half_seq_len:]
        sequences[i, :seq_len, 2] = 1. - ramp
    return sequences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepRC 2020 Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True)
    parser.add_argument('--k', type=str, required=True,
                        help='Comma-separated k values, e.g. 100,1000,10000')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--n_updates', type=int, default=int(1e4))
    parser.add_argument('--evaluate_at', type=int, default=int(1e3))
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results/deeprc')
    parser.add_argument('--max_repertoires', type=int, default=None,
                        help='Subsample to this many repertoires total (for debug runs).')
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help='results_dir from disease classification run; '
                             'driver eval loads from its checkpoints instead of retraining.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_n_sequences', type=int, default=int(1e4))
    parser.add_argument('--max_seq_len', type=int, default=50)
    args = parser.parse_args()
    ks = [int(x) for x in args.k.split(',')]

    evaluator = DeepRC2020DriverIdentificationEvaluator(
        n_updates=args.n_updates,
        evaluate_at=args.evaluate_at,
        device=args.device,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        sample_n_sequences=args.sample_n_sequences,
        max_seq_len=args.max_seq_len,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        ks=ks,
        max_repertoires=args.max_repertoires,
        model_save_dir=args.model_save_dir,
        raw_file_cache={},
        output_csv=args.output_csv,
    )
