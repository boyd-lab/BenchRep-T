"""
Driver sequence identification evaluation for the Ostmeyer 2019 MIL-TCR model.

Per-sequence scores are derived from the trained logistic regression applied to
Atchley-encoded 4-mers: for each CDR3, extract all 4-mers from the core region
(positions 3..-3), look up each 4-mer's instance score (sigmoid of the logit),
and assign the CDR3 the maximum score across its 4-mers.  CDR3s whose 4-mers
are all absent from the training-derived 4-mer score map receive score 0.5
(neutral).

Sequences are ranked descending by score and the top-k CDR3s are compared
against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.ostmeyer_2019 import MIL_TCR_Classifier


class Ostmeyer2019DriverIdentificationEvaluator:
    """
    Evaluates Ostmeyer 2019's ability to identify disease-associated driver
    sequences using per-sequence max-4mer logistic scores.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, train_val_ratio=0.9, n_restarts=200, lbfgsb_maxiter=1000,
                 abundance_method='A',
                 sequence_col='cdr3_aa',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None):
        self.train_val_ratio = train_val_ratio
        self.n_restarts = n_restarts
        self.lbfgsb_maxiter = lbfgsb_maxiter
        self.abundance_method = abundance_method
        self.sequence_col = sequence_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map

    # ------------------------------------------------------------------
    # Metadata helpers (mirrors disease classification evaluator)
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

    def _make_model(self, abundance_method=None):
        method = abundance_method if abundance_method is not None else self.abundance_method
        return MIL_TCR_Classifier(
            n_restarts=self.n_restarts,
            lbfgsb_maxiter=self.lbfgsb_maxiter,
            abundance_method=method,
            sequence_col=self.sequence_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n,
            indices_map=self.indices_map,
        )

    def _tune_abundance_method(self, all_files, train_files, train_labels,
                               val_files, val_labels,
                               candidates=('A', 'B')):
        """Tune abundance method via validation AUROC."""
        best_method, best_auroc = candidates[0], -1.0

        for method in candidates:
            model = self._make_model(method)
            model.preload_repertoires(all_files)
            for fp in all_files:
                model.extract_4mer_features(fp, use_cache=True)

            try:
                model.train(train_files, train_labels)
            except Exception as e:
                print(f"  method={method}: Training failed: {e}")
                continue

            val_probs = np.array([
                model.predict_diagnosis(fp)['probability_positive']
                for fp in val_files
            ])
            auroc = roc_auc_score(np.array(val_labels), val_probs)
            print(f"  method={method}: Val AUROC={auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_method = method

        print(f"  Best abundance method: {best_method} (Val AUROC={best_auroc:.4f})")
        return best_method

    # ------------------------------------------------------------------
    # Per-sequence scoring
    # ------------------------------------------------------------------

    def score_repertoire_cdr3s(self, file_path, model):
        """
        Score each unique CDR3 in a repertoire by its maximum 4-mer logistic score.

        The trained model weights are applied to Atchley-encoded 4-mers to produce
        per-4mer instance probabilities (sigmoid of logit).  Each CDR3 receives the
        maximum score across all of its 4-mers.  CDR3s with no valid 4-mers
        receive the neutral score 0.5.

        Returns:
            List of (cdr3, score) sorted descending by score.
        """
        if model.weights is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = model.load_repertoire(file_path)
        if df.empty or model.sequence_col not in df.columns:
            return []

        # Get the 4-mer score map for this repertoire (uses cached features)
        if file_path in model._features_cache:
            features, log_abundances, fourmer_strings = model._features_cache[file_path]
        else:
            features, log_abundances, fourmer_strings = model._build_fourmer_data(file_path)

        if len(features) == 0:
            return []

        # Per-4mer instance scores
        instance_scores = model._compute_instance_scores(features, log_abundances)
        fourmer_score_map = {fm: float(s)
                             for fm, s in zip(fourmer_strings, instance_scores)}

        # Per-CDR3 score: max over its 4-mers
        cdr3_scores = {}
        for cdr3 in df[model.sequence_col].dropna().unique():
            cdr3 = str(cdr3)
            core = cdr3[3:-3]  # exclude first 3 and last 3 residues
            best = 0.5
            for i in range(len(core) - 3):
                fm = core[i:i + 4]
                score = fourmer_score_map.get(fm)
                if score is not None and score > best:
                    best = score
            cdr3_scores[cdr3] = best

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
                             n_folds=3, random_state=7,
                             abundance_method_candidates=None,
                             allowed_participants=None,
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification.

        For each fold:
        1. Tune abundance method on a train/val split.
        2. Train final model on all non-test data.
        3. Score each test CDR3 and compare top-k against ground truth.

        Returns:
            DataFrame with per-repertoire precision@k and recall@k.
        """
        if abundance_method_candidates is None:
            abundance_method_candidates = ['A', 'B']

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

        all_results = []
        fold_summaries = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            test_mask = metadata[fold_col] == test_fold
            train_val_data = metadata[~test_mask]
            test_data = metadata[test_mask]

            train_data, val_data = train_test_split(
                train_val_data,
                train_size=self.train_val_ratio,
                random_state=random_state,
                stratify=train_val_data['label'],
            )

            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()
            val_files = val_data['file_path'].tolist()
            val_labels = val_data['label'].tolist()
            train_val_files = train_val_data['file_path'].tolist()
            train_val_labels = train_val_data['label'].tolist()
            test_files = test_data['file_path'].tolist()

            print(f"Train: {len(train_data)}, Val: {len(val_data)}, "
                  f"Test: {len(test_data)}")

            all_files = train_val_files + test_files

            # --- Tune abundance method ---
            print("\n--- Tuning abundance method ---")
            best_method = self._tune_abundance_method(
                all_files, train_files, train_labels,
                val_files, val_labels,
                candidates=abundance_method_candidates,
            )

            # --- Train final model on all non-test data ---
            print(f"\nTraining final model (method={best_method}) on "
                  f"{len(train_val_files)} samples...")
            final_model = self._make_model(best_method)
            final_model.preload_repertoires(all_files)
            for fp in all_files:
                final_model.extract_4mer_features(fp, use_cache=True)
            final_model.train(train_val_files, train_val_labels)

            # --- Score test repertoires ---
            print(f"\n--- Scoring test repertoires (k={k}) ---")
            fold_precisions = []
            fold_recalls = []

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Scoring"):
                file_path = row['file_path']
                filename_stem = os.path.basename(file_path) \
                    .replace('.tsv.gz', '').replace('.tsv', '')

                if filename_stem not in drivers_by_file:
                    continue

                driver_cdr3s = drivers_by_file[filename_stem]
                ranked = self.score_repertoire_cdr3s(file_path, final_model)
                top_k_cdr3s = set(cdr3 for cdr3, _ in ranked[:k])

                hits = top_k_cdr3s & driver_cdr3s
                precision = len(hits) / k
                recall = len(hits) / len(driver_cdr3s)

                fold_precisions.append(precision)
                fold_recalls.append(recall)

                all_results.append({
                    'fold': test_fold,
                    'filename': filename_stem,
                    'participant_label': row[participant_col],
                    'disease_label': int(row['label']),
                    'n_repertoire_unique_cdr3s': len(ranked),
                    'n_ground_truth_drivers': len(driver_cdr3s),
                    'n_hits_at_k': len(hits),
                    'precision_at_k': precision,
                    'recall_at_k': recall,
                    'best_abundance_method': best_method,
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

            final_model.clear_cache()

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
            micro_precision = total_hits / (len(results_df) * k) if len(results_df) > 0 else 0.0
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ostmeyer 2019 Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--n_restarts', type=int, default=200)
    parser.add_argument('--abundance_method_candidates', type=str, nargs='+',
                        default=['A', 'B'])
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--random_state', type=int, default=7)
    args = parser.parse_args()

    evaluator = Ostmeyer2019DriverIdentificationEvaluator(
        train_val_ratio=args.train_val_ratio,
        n_restarts=args.n_restarts,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        k=args.k,
        random_state=args.random_state,
        abundance_method_candidates=args.abundance_method_candidates,
        output_csv=args.output_csv,
    )
