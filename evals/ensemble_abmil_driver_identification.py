"""
Driver sequence identification evaluation for the ABMIL model.

Per-sequence importance is the gated attention weight assigned by the trained
TCRGatedAttentionMIL network.  After softmax across all sequences in a
repertoire, higher attention weight indicates the model relied more heavily on
that sequence when predicting disease status.

Multiple sequences sharing the same CDR3 (but differing V/J genes) are
aggregated by taking the maximum attention weight across all matching rows.

Sequences are ranked descending by attention weight and the top-k CDR3s are
compared against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.ensemble_abmil import ABMIL


class ABMILDriverIdentificationEvaluator:
    """
    Evaluates ABMIL's ability to identify disease-associated driver sequences
    using per-sequence gated attention weights.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, seed=7,
                 sequence_col='cdr3_aa', v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None, ignore_allele=False,
                 max_instances=10000, M=128, L=64,
                 epochs=100, lr=5e-4, weight_decay=1e-4, patience=10,
                 use_gpu=True, dropout=0.25, max_length=40,
                 embedding_dim_aa=64, embedding_dim_genes=48, kernel=5,
                 conv_units=(32, 64, 128), features='full'):
        self.val_split = val_split
        self.seed = seed
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.ignore_allele = ignore_allele
        self.max_instances = max_instances
        self.M = M
        self.L = L
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_gpu = use_gpu
        self.dropout = dropout
        self.max_length = max_length
        self.embedding_dim_aa = embedding_dim_aa
        self.embedding_dim_genes = embedding_dim_genes
        self.kernel = kernel
        self.conv_units = tuple(conv_units)
        self.features = features

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
    # Model helpers
    # ------------------------------------------------------------------

    def _make_model(self):
        return ABMIL(
            max_instances=self.max_instances,
            M=self.M, L=self.L,
            epochs=self.epochs, lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            val_split=self.val_split,
            seed=self.seed,
            sequence_col=self.sequence_col,
            v_gene_col=self.v_gene_col,
            j_gene_col=self.j_gene_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n,
            indices_map=self.indices_map,
            ignore_allele=self.ignore_allele,
            use_gpu=self.use_gpu,
            dropout=self.dropout,
            max_length=self.max_length,
            embedding_dim_aa=self.embedding_dim_aa,
            embedding_dim_genes=self.embedding_dim_genes,
            kernel=self.kernel,
            conv_units=self.conv_units,
            features=self.features,
        )

    # ------------------------------------------------------------------
    # Per-sequence scoring
    # ------------------------------------------------------------------

    def score_repertoire_cdr3s(self, file_path, model):
        """
        Score each sequence in a repertoire using gated attention weights.

        The gated attention mechanism (tanh × sigmoid branches) produces a
        pre-softmax score per sequence; after softmax these are the attention
        weights used to pool sequences into the bag embedding.  Higher weight
        means the model relied more on that sequence for the prediction.

        Multiple rows with the same CDR3 receive the max attention weight
        across those rows.

        Returns:
            List of (cdr3, attention_weight) sorted descending.
        """
        model.encoder.eval()
        model.model.eval()

        df = model.load_repertoire(file_path, apply_subsampling=False)
        if len(df) == 0:
            return []

        seq_arr, v_arr, j_arr = model._get_per_seq_arrays(file_path, apply_subsampling=False)
        if seq_arr.shape[0] == 0:
            return []

        seq_t, v_t, j_t = model._to_tensors(seq_arr, v_arr, j_arr)
        with torch.no_grad():
            H = model.encoder(seq_t, v_t, j_t)                          # (K, output_dim)
            H_enc = model.model.encoder(H)                               # (K, M)
            A = model.model.attention_w(
                model.model.attention_V(H_enc) * model.model.attention_U(H_enc)
            )                                                            # (K, 1)
            A = F.softmax(A.squeeze(dim=1), dim=0).cpu().numpy()         # (K,)

        cdr3s = df[model.sequence_col].fillna('').tolist()

        # Aggregate by CDR3 — take max attention weight across identical CDR3s
        cdr3_scores = {}
        for cdr3, attn in zip(cdr3s, A):
            cdr3_scores[cdr3] = max(cdr3_scores.get(cdr3, 0.0), float(attn))

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
                             n_folds=3, random_state=7,
                             allowed_participants=None,
                             max_repertoires=None,
                             model_save_dir=None,
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification.

        For each fold:
        1. Train ABMIL on all non-test data (internal val split for early stopping).
        2. Extract per-sequence gated attention weights for each test repertoire.
        3. Compare top-k CDR3s against ground truth driver sequences for each k in ks.

        Returns:
            DataFrame with per-repertoire, per-k precision@k and recall@k.
        """
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
            metadata = metadata.sample(n=max_repertoires, random_state=self.seed)
            print(f"Subsampled to {max_repertoires} repertoires for debug run.")

        drivers_by_file = self.load_driver_sequences(driver_seqs_path, target_disease)

        all_results = []
        fold_summaries = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            test_mask = metadata[fold_col] == test_fold
            train_val_data = metadata[~test_mask]
            test_data = metadata[test_mask]

            train_files = train_val_data['file_path'].tolist()
            train_labels = train_val_data['label'].tolist()

            print(f"Train: {len(train_val_data)}, Test: {len(test_data)}")

            fold_save_dir = (os.path.join(model_save_dir, target_disease, f'fold{test_fold}')
                             if model_save_dir else None)
            if fold_save_dir and os.path.isdir(fold_save_dir):
                print(f"  Loading model from checkpoint: {fold_save_dir}")
                model = ABMIL.load(fold_save_dir, use_gpu=self.use_gpu)
            else:
                model = self._make_model()
                train_result = model.train(train_files, train_labels)
                print(f"  Best val loss: {train_result['best_val_loss']:.4f}, "
                      f"Epochs: {train_result['epochs_trained']}")

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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABMIL Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True)
    parser.add_argument('--k', type=str, required=True,
                        help='Comma-separated k values, e.g. 100,1000,10000')
    parser.add_argument('--model_save_dir', type=str, default=None,
                        help='Directory written by --model_save_dir in '
                             'ensemble_abmil_disease_classification.py')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--features', type=str, default='full',
                        choices=['full', 'cdr3_only', 'vj_only'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_repertoires', type=int, default=None,
                        help='Subsample to this many repertoires total (for debug runs).')
    parser.add_argument('--random_state', type=int, default=7)
    args = parser.parse_args()
    ks = [int(x) for x in args.k.split(',')]

    evaluator = ABMILDriverIdentificationEvaluator(
        features=args.features,
        epochs=args.epochs,
        patience=args.patience,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        ks=ks,
        max_repertoires=args.max_repertoires,
        random_state=args.random_state,
        model_save_dir=args.model_save_dir,
        output_csv=args.output_csv,
    )
