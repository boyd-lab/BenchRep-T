"""
Driver sequence identification evaluation for DeepTCR (Sidhom et al. 2021).

Per-sequence importance is derived from DeepTCR's singleton-bag inference:
after training, each test sequence is scored by running the trained network
with a single-item bag (that sequence alone), giving the predicted disease
probability as if that sequence were the sole evidence.  This directly uses
the concept-attention aggregation mechanism — a sequence that by itself
activates disease-associated concepts receives a high score.

Internally, `_train()` calls `Get_Sequence_Pred`, which constructs a diagonal
sparse matrix so every sequence is evaluated as an independent bag, and stores
the result in `dtcr.predicted[i, disease_class_idx]`.

Multiple rows sharing the same CDR3 (but differing V/J genes) are aggregated
by taking the maximum singleton-bag probability across all matching rows.

Sequences are ranked descending by predicted disease probability and the top-k
CDR3s are compared against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.deeptcr_2021_disease_classification import DeepTCREvaluator
from Layers import make_test_pred_object
from utils_s import Get_Train_Valid_Test_KFold

from sklearn.model_selection import train_test_split


class DeepTCRDriverIdentificationEvaluator(DeepTCREvaluator):
    """
    Evaluates DeepTCR's ability to identify disease-associated driver sequences
    using per-sequence singleton-bag predicted disease probability.
    """

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
    # Per-sequence scoring
    # ------------------------------------------------------------------

    def score_repertoire_cdr3s(self, specimen, dtcr, disease_class_idx):
        """
        Extract per-CDR3 singleton-bag disease probabilities for a specimen.

        `dtcr.predicted` is populated by Get_Sequence_Pred inside _train(),
        which treats each test sequence as a singleton bag (diagonal sparse
        matrix), so dtcr.predicted[i, disease_class_idx] is the probability
        the network assigns when sequence i is the only member of the bag.

        Multiple rows with the same CDR3 receive the max probability across
        all matching rows.

        Returns:
            List of (cdr3, probability) sorted descending.
        """
        mask = dtcr.sample_id == specimen
        if not np.any(mask):
            return []

        cdr3s = dtcr.beta_sequences[mask]
        scores = dtcr.predicted[mask, disease_class_idx]

        cdr3_scores = {}
        for cdr3, score in zip(cdr3s, scores):
            if cdr3 is None:
                continue
            cdr3_scores[cdr3] = max(cdr3_scores.get(cdr3, 0.0), float(score))

        return sorted(cdr3_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Main cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                             driver_seqs_path, k,
                             participant_col='participant_label',
                             file_prefix='part_table_', file_suffix='.tsv.gz',
                             disease_col='disease',
                             fold_col='malid_cross_validation_fold_id_when_in_test_set',
                             n_folds=3,
                             random_state=None,
                             allowed_participants=None,
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification using DeepTCR.

        For each fold:
        1. Train DeepTCR_WF on all non-test specimens.
        2. Extract per-sequence singleton-bag disease probabilities for each
           test specimen from dtcr.predicted.
        3. Compare top-k CDR3s against ground truth driver sequences.

        Returns:
            DataFrame with per-repertoire precision@k and recall@k.
        """
        # DeepTCR sources must be on sys.path — done in DeepTCREvaluator import
        from DeepTCR import DeepTCR_WF

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

        drivers_by_file = self.load_driver_sequences(driver_seqs_path, target_disease)

        print("\nLoading all repertoire files into memory...")
        beta_sequences, sample_labels, class_labels, counts, v_beta, j_beta, written = \
            self._collect_all_data(metadata, target_disease)

        if beta_sequences is None:
            print("Error: no sequences could be loaded.")
            return pd.DataFrame()

        print(f"Loaded {len(beta_sequences):,} sequences from {len(written)} specimens.")

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
            print(f"Train: {len(train_data)}, Val: {len(val_data)}, "
                  f"Test: {len(test_data)}")

            fold_name = os.path.join(
                self.results_dir, f"{target_disease}_driver_fold{test_fold}"
            )
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
                idx = [sample_name_to_idx[s]
                       for s in specimens['specimen_label']
                       if s in sample_name_to_idx]
                return np.array(idx, dtype=int)

            test_idx = _get_indices(test_data)
            train_val_idx = _get_indices(
                pd.concat([train_data, val_data], ignore_index=True)
            )

            if len(test_idx) == 0:
                print(f"Warning: no test samples found for fold {test_fold}, skipping.")
                fold_summaries.append({
                    'fold': test_fold, 'n_repertoires': 0,
                    'mean_precision_at_k': float('nan'),
                    'mean_recall_at_k': float('nan'),
                })
                continue

            train_idx, val_idx = train_test_split(
                train_val_idx,
                train_size=self.train_val_ratio,
                random_state=rng,
                stratify=Y[train_val_idx].argmax(axis=1),
            )

            dtcr.train, dtcr.valid, dtcr.test = Get_Train_Valid_Test_KFold(
                Vars=Vars,
                train_idx=train_idx,
                valid_idx=val_idx,
                test_idx=test_idx,
                Y=Y,
            )
            dtcr.LOO = None

            dtcr._reset_models()
            dtcr._build(
                kernel=self.kernel,
                num_concepts=self.num_concepts,
                size_of_net=self.size_of_net,
                epochs_min=self.epochs_min,
                epochs_max=self.epochs_max,
                hinge_loss_t=self.hinge_loss_t,
                train_loss_min=None,
                convergence='validation',
                batch_size=self.batch_size,
                suppress_output=False,
            )
            dtcr.test_pred = make_test_pred_object()
            # write=False: we only need dtcr.predicted (populated regardless of write flag)
            dtcr._train(write=False, batch_seed=None, iteration=test_fold)

            print(f"\n--- Scoring test repertoires (k={k}) ---")
            fold_precisions = []
            fold_recalls = []

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Scoring"):
                file_path = row['file_path']
                filename_stem = (
                    os.path.basename(file_path)
                    .replace('.tsv.gz', '').replace('.tsv', '')
                )
                specimen = row['specimen_label']

                if filename_stem not in drivers_by_file:
                    continue

                driver_cdr3s = drivers_by_file[filename_stem]
                ranked = self.score_repertoire_cdr3s(specimen, dtcr, disease_class_idx)
                top_k_cdr3s = set(cdr3 for cdr3, _ in ranked[:k])

                hits = top_k_cdr3s & driver_cdr3s
                precision = len(hits) / k
                recall = len(hits) / len(driver_cdr3s)

                fold_precisions.append(precision)
                fold_recalls.append(recall)

                all_results.append({
                    'fold': test_fold,
                    'filename': filename_stem,
                    'specimen_label': specimen,
                    'participant_label': row[participant_col],
                    'disease_label': int(row['label']),
                    'n_repertoire_unique_cdr3s': len(ranked),
                    'n_ground_truth_drivers': len(driver_cdr3s),
                    'n_hits_at_k': len(hits),
                    'precision_at_k': precision,
                    'recall_at_k': recall,
                })

            if fold_precisions:
                mean_prec = np.mean(fold_precisions)
                mean_rec = np.mean(fold_recalls)
                print(f"\nFold {test_fold}: {len(fold_precisions)} repertoires")
                print(f"  Mean Precision@{k}: {mean_prec:.4f}")
                print(f"  Mean Recall@{k}:    {mean_rec:.4f}")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': len(fold_precisions),
                    'mean_precision_at_k': mean_prec,
                    'mean_recall_at_k': mean_rec,
                })
            else:
                print(f"\nFold {test_fold}: No test repertoires with ground truth drivers")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': 0,
                    'mean_precision_at_k': float('nan'),
                    'mean_recall_at_k': float('nan'),
                })

            del dtcr

        results_df = pd.DataFrame(all_results)

        if len(results_df) > 0:
            overall_precision = results_df['precision_at_k'].mean()
            overall_recall = results_df['recall_at_k'].mean()
            total_hits = results_df['n_hits_at_k'].sum()
            total_possible = results_df['n_ground_truth_drivers'].sum()
        else:
            overall_precision = overall_recall = 0.0
            total_hits = total_possible = 0

        print(f"\n{'=' * 60}")
        print(f"OVERALL RESULTS: {target_disease} Driver Identification (k={k})")
        print(f"{'=' * 60}")
        print(f"Repertoires evaluated: {len(results_df)}")
        print(f"Overall Precision@{k} (macro): {overall_precision:.4f}")
        print(f"Overall Recall@{k}    (macro): {overall_recall:.4f}")
        if total_possible > 0:
            micro_recall = total_hits / total_possible
            micro_precision = (
                total_hits / (len(results_df) * k) if len(results_df) > 0 else 0.0
            )
            print(f"Overall Precision@{k} (micro): {micro_precision:.4f}")
            print(f"Overall Recall@{k}    (micro): {micro_recall:.4f}")

        print(f"\nPer-fold breakdown:")
        for s in fold_summaries:
            print(f"  Fold {s['fold']}: {s['n_repertoires']} reps, "
                  f"P@{k}={s['mean_precision_at_k']:.4f}, "
                  f"R@{k}={s['mean_recall_at_k']:.4f}")

        if output_csv and len(results_df) > 0:
            results_df.to_csv(output_csv, index=False)
            print(f"\nPer-repertoire results saved to: {output_csv}")

        return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepTCR Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results/deeptcr_driver')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--num_concepts', type=int, default=64)
    parser.add_argument('--size_of_net', type=str, default='small',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs_min', type=int, default=10)
    parser.add_argument('--epochs_max', type=int, default=None)
    parser.add_argument('--hinge_loss_t', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--random_state', type=int, default=7)
    args = parser.parse_args()

    evaluator = DeepTCRDriverIdentificationEvaluator(
        kernel=args.kernel,
        num_concepts=args.num_concepts,
        size_of_net=args.size_of_net,
        epochs_min=args.epochs_min,
        epochs_max=args.epochs_max,
        hinge_loss_t=args.hinge_loss_t,
        batch_size=args.batch_size,
        device=args.device,
        results_dir=args.results_dir,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        k=args.k,
        random_state=args.random_state,
        output_csv=args.output_csv,
    )
