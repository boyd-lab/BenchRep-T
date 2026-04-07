"""
Driver sequence identification evaluation for Emerson 2017 model.

Uses the Fisher's exact test p-values (computed during training) as
per-sequence scores to rank CDR3s within each test repertoire, then
compares the top-k predictions against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.

Ground truth format (CSV):
    disease, sample_cdr3, sample_vgene, sample_jgene, ..., filename
    where ``filename`` is the repertoire stem (e.g. part_table_BFI-..._S001)
    and ``sample_cdr3`` is the driver CDR3 amino-acid sequence.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models.emerson_2017 import CMV_Immunosequencing_Model


class Emerson2017DriverIdentificationEvaluator:
    """
    Evaluates Emerson 2017's ability to identify disease-associated driver
    sequences using Fisher's exact test p-values as sequence-level scores.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, train_val_ratio=0.9,
                 sequence_col='cdr3_aa', v_col='v_call', j_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None):
        self.train_val_ratio = train_val_ratio
        self.sequence_col = sequence_col
        self.v_col = v_col
        self.j_col = j_col
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
        """
        Load ground truth driver CDR3s grouped by repertoire filename.

        Returns:
            dict: {filename_stem -> set of CDR3 strings}
        """
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

    def _make_model(self, p_value_threshold=1e-4):
        return CMV_Immunosequencing_Model(
            p_value_threshold=p_value_threshold,
            sequence_col=self.sequence_col,
            v_col=self.v_col,
            j_col=self.j_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n,
            indices_map=self.indices_map,
        )

    def _tune_p_value(self, base_model, base_files, base_labels,
                      val_files, val_labels, p_value_candidates):
        """Tune p-value threshold via validation-set AUROC (same as disease
        classification) and return the best threshold."""
        best_p, best_auroc = p_value_candidates[0], -1.0

        for p_val in p_value_candidates:
            model = self._make_model(p_value_threshold=p_val)
            model._repertoire_cache = base_model._repertoire_cache
            model._tcr_stats_cache = base_model._tcr_stats_cache
            model.select_diagnostic_tcrs_from_cache(p_val)

            if len(model.diagnostic_tcrs) == 0:
                print(f"  p={p_val:.0e}: No diagnostic TCRs, skipping")
                continue

            model.train_beta_binomial_model(base_files, base_labels)
            val_probs = np.array([
                model.predict_diagnosis(fp)['probability_positive']
                for fp in val_files
            ])
            val_labels_arr = np.array(val_labels)
            auroc = roc_auc_score(val_labels_arr, val_probs)
            print(f"  p={p_val:.0e}: {len(model.diagnostic_tcrs)} TCRs, "
                  f"Val AUROC={auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_p = p_val

        print(f"  Best p-value: {best_p:.0e} (Val AUROC={best_auroc:.4f})")
        return best_p

    # ------------------------------------------------------------------
    # Sequence scoring
    # ------------------------------------------------------------------

    def score_repertoire_cdr3s(self, file_path, model, tcr_pvalues):
        """
        Score each unique CDR3 in a repertoire by its best (lowest) Fisher
        p-value across all (V, CDR3, J) tuples.

        Sequences absent from the training statistics receive p-value = 1.0.

        Returns:
            List of (cdr3, p_value) sorted ascending by p-value.
        """
        repertoire_seqs = model.load_repertoire(file_path)

        cdr3_best_pval = {}
        for (v, cdr3, j) in repertoire_seqs:
            pval = tcr_pvalues.get((v, cdr3, j), 1.0)
            if cdr3 not in cdr3_best_pval or pval < cdr3_best_pval[cdr3]:
                cdr3_best_pval[cdr3] = pval

        return sorted(cdr3_best_pval.items(), key=lambda x: x[1])

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
                             p_value_candidates=None,
                             allowed_participants=None,
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification.

        For each fold:
        1. Tune p-value threshold on a train/val split (same as disease
           classification for consistency).
        2. Recompute TCR statistics on ALL training data with the best
           threshold to obtain final per-TCR p-values.
        3. Score each test repertoire's CDR3s and compare the top-k against
           ground truth driver sequences.

        Args:
            metadata_path: Path to metadata.tsv
            target_disease: Disease name (e.g. 'Covid19', 'HIV', 'Influenza')
            data_dir: Directory containing repertoire files
            driver_seqs_path: Path to ground truth driver sequences CSV
            k: Number of top-ranked CDR3s to consider
            output_csv: Optional path to save per-repertoire results

        Returns:
            DataFrame with per-repertoire precision@k and recall@k.
        """
        if p_value_candidates is None:
            p_value_candidates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        # --- Load and prepare data ---
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

        # Load ground truth
        drivers_by_file = self.load_driver_sequences(driver_seqs_path, target_disease)

        all_results = []
        fold_summaries = []

        for test_fold in range(n_folds):
            print(f"\n{'=' * 60}")
            print(f"FOLD {test_fold}")
            print(f"{'=' * 60}")

            # --- Split ---
            test_mask = metadata[fold_col] == test_fold
            train_val_data = metadata[~test_mask]
            test_data = metadata[test_mask]

            # Train/val split for p-value tuning
            base_data, val_data = train_test_split(
                train_val_data, train_size=self.train_val_ratio,
                random_state=random_state,
                stratify=train_val_data['label'],
            )

            base_files = base_data['file_path'].tolist()
            base_labels = base_data['label'].tolist()
            val_files = val_data['file_path'].tolist()
            val_labels = val_data['label'].tolist()
            train_val_files = train_val_data['file_path'].tolist()
            train_val_labels = train_val_data['label'].tolist()
            test_files = test_data['file_path'].tolist()

            print(f"Base: {len(base_data)}, Val: {len(val_data)}, "
                  f"Test: {len(test_data)}")

            # --- Preload ALL repertoires ---
            all_files = train_val_files + test_files
            base_model = self._make_model()
            base_model.preload_repertoires(all_files)

            # --- Tune p-value threshold on base/val split ---
            print("\n--- Tuning p-value threshold ---")
            base_model.compute_tcr_statistics(base_files, base_labels)
            best_p = self._tune_p_value(
                base_model, base_files, base_labels,
                val_files, val_labels, p_value_candidates,
            )

            # --- Recompute statistics on ALL training data ---
            print(f"\nRecomputing TCR statistics on all "
                  f"{len(train_val_files)} training samples...")
            final_model = self._make_model(p_value_threshold=best_p)
            final_model._repertoire_cache = base_model._repertoire_cache
            final_model.compute_tcr_statistics(train_val_files, train_val_labels)

            tcr_pvalues = final_model._tcr_stats_cache['tcr_pvalues']
            print(f"Total scored TCRs: {len(tcr_pvalues)}")

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

                # Rank CDR3s by p-value (ascending)
                ranked = self.score_repertoire_cdr3s(
                    file_path, final_model, tcr_pvalues)
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
                    'best_p_value_threshold': best_p,
                })

            # --- Fold summary ---
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
                    'best_p_value': best_p,
                })
            else:
                print(f"\nFold {test_fold}: No test repertoires with "
                      f"ground truth drivers")
                fold_summaries.append({
                    'fold': test_fold,
                    'n_repertoires': 0,
                    'mean_precision_at_k': float('nan'),
                    'mean_recall_at_k': float('nan'),
                    'best_p_value': best_p,
                })

            # Free memory between folds
            final_model.clear_cache()
            base_model.clear_cache()

        # ------------------------------------------------------------------
        # Overall results
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(all_results)

        if len(results_df) > 0:
            overall_precision = results_df['precision_at_k'].mean()
            overall_recall = results_df['recall_at_k'].mean()
            total_hits = results_df['n_hits_at_k'].sum()
            total_possible = results_df['n_ground_truth_drivers'].sum()
        else:
            overall_precision = 0.0
            overall_recall = 0.0
            total_hits = 0
            total_possible = 0

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
                  f"R@{k}={s['mean_recall_at_k']:.4f}, "
                  f"best_p={s['best_p_value']:.0e}")

        if output_csv and len(results_df) > 0:
            results_df.to_csv(output_csv, index=False)
            print(f"\nPer-repertoire results saved to: {output_csv}")

        return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emerson 2017 Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to evaluate (e.g. Covid19, HIV, Influenza)')
    parser.add_argument('--driver_seqs_path', type=str, required=True,
                        help='Path to ground truth driver sequences CSV')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of top-ranked CDR3s to consider')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-repertoire results CSV')
    parser.add_argument('--p_value_candidates', type=float, nargs='+',
                        default=None,
                        help='P-value thresholds to try '
                             '(default: 1e-2 1e-3 1e-4 1e-5 1e-6)')
    parser.add_argument('--train_val_ratio', type=float, default=0.9,
                        help='Train/val ratio for threshold tuning (default: 0.9)')
    parser.add_argument('--random_state', type=int, default=7,
                        help='Random seed (default: 7)')
    args = parser.parse_args()

    evaluator = Emerson2017DriverIdentificationEvaluator(
        train_val_ratio=args.train_val_ratio,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        k=args.k,
        random_state=args.random_state,
        p_value_candidates=args.p_value_candidates,
        output_csv=args.output_csv,
    )
