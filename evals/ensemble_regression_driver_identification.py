"""
Driver sequence identification evaluation for the Ensemble Regression model.

Derives per-sequence scores from the trained logistic regression coefficients:
  - K-mer sub-model: for each CDR3, extract its gapped 4-mers, pass through the
    fitted vectorizer + scaler, and compute the decision function (log-odds).
  - V/J sub-model: for each (V, J) pair, build a single-count feature vector,
    pass through the fitted vectorizer + scaler, and compute the decision function.
  - Ensemble: alpha * kmer_score + (1 - alpha) * vj_score

Sequences are ranked by score (descending), and the top-k CDR3s are compared
against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.ensemble_regression import Gapped_4mer_VJgene, _extract_kmers


class EnsembleRegressionDriverIdentificationEvaluator:
    """
    Evaluates Ensemble Regression's ability to identify disease-associated
    driver sequences using per-sequence logistic regression scores.
    """

    HEALTHY_LABEL = "Healthy/Background"

    def __init__(self, val_split=0.2, n_cv_folds=5,
                 sequence_col='cdr3_aa', v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None, submodel='ensemble'):
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.submodel = submodel

    # ------------------------------------------------------------------
    # Metadata helpers (shared pattern across evaluators)
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

    def _make_model(self):
        return Gapped_4mer_VJgene(
            val_split=self.val_split,
            n_cv_folds=self.n_cv_folds,
            sequence_col=self.sequence_col,
            v_gene_col=self.v_gene_col,
            j_gene_col=self.j_gene_col,
            subsample_fraction=self.subsample_fraction,
            subsample_seed=self.subsample_seed,
            subsample_n=self.subsample_n,
            indices_map=self.indices_map,
            submodel=self.submodel,
        )

    # ------------------------------------------------------------------
    # Per-sequence scoring
    # ------------------------------------------------------------------

    def score_repertoire_cdr3s(self, file_path, model):
        """
        Score each unique CDR3 in a repertoire using the trained model's
        logistic regression coefficients.

        For each unique (V, CDR3, J) triple in the repertoire:
          - K-mer score: decision_function on the sequence's gapped 4-mer features
          - V/J score: decision_function on the sequence's V/J gene features
          - Combined: alpha * kmer_score + (1 - alpha) * vj_score

        Multiple triples sharing the same CDR3 are aggregated by taking the
        maximum score.

        Returns:
            List of (cdr3, score) sorted descending by score.
        """
        df = model.load_repertoire(file_path)

        seq_col = model.sequence_col
        v_col = model.v_gene_col
        j_col = model.j_gene_col

        # Unique (V, CDR3, J) triples
        triples_df = df[[v_col, seq_col, j_col]].dropna().drop_duplicates()

        if len(triples_df) == 0:
            return []

        use_kmer = model.submodel in ('ensemble', 'kmer_only')
        use_vj = model.submodel in ('ensemble', 'vj_only')

        scores = np.zeros(len(triples_df))

        # --- K-mer sub-model ---
        if use_kmer:
            kmer_dicts = []
            for cdr3 in triples_df[seq_col].values:
                counts = {}
                for kmer in _extract_kmers(str(cdr3), model.kmer_size, model.use_gaps):
                    counts[kmer] = counts.get(kmer, 0) + 1
                kmer_dicts.append(counts)

            X_kmer = model.kmer_scaler.transform(
                model.kmer_vectorizer.transform(kmer_dicts))
            kmer_scores = model.kmer_model.decision_function(X_kmer)

            if model.submodel == 'ensemble':
                scores += model.best_alpha * kmer_scores
            else:
                scores = kmer_scores

        # --- V/J sub-model ---
        if use_vj:
            vj_dicts = []
            v_values = triples_df[v_col].values
            j_values = triples_df[j_col].values
            for v, j in zip(v_values, j_values):
                vj_dict = {}
                v_norm = model._normalize_gene(v)
                j_norm = model._normalize_gene(j)
                vj_dict[f'V:{v_norm}'] = 1
                vj_dict[f'J:{j_norm}'] = 1
                vj_dicts.append(vj_dict)

            X_vj = model.vj_scaler.transform(
                model.vj_vectorizer.transform(vj_dicts))
            vj_scores = model.vj_model.decision_function(X_vj)

            if model.submodel == 'ensemble':
                scores += (1 - model.best_alpha) * vj_scores
            else:
                scores = vj_scores

        # --- Aggregate to CDR3 level (max score per unique CDR3) ---
        triples_df = triples_df.copy()
        triples_df['score'] = scores
        cdr3_scores = triples_df.groupby(seq_col)['score'].max()

        # Sort descending (higher score = more disease-associated)
        ranked = cdr3_scores.sort_values(ascending=False)
        return list(zip(ranked.index, ranked.values))

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
                             output_csv=None):
        """
        Run k-fold CV for driver sequence identification.

        For each fold:
        1. Train model on all non-test data (internal tuning handled by
           model.train()).
        2. Derive per-sequence scores from the trained logistic regression
           coefficients.
        3. Compare top-k CDR3s against ground truth driver sequences for each
           k value in ``ks``.

        Args:
            metadata_path: Path to metadata.tsv
            target_disease: Disease name (e.g. 'Covid19', 'HIV', 'Influenza')
            data_dir: Directory containing repertoire files
            driver_seqs_path: Path to ground truth driver sequences CSV
            ks: Iterable of int top-ranked CDR3 cutoffs to evaluate
            output_csv: Optional path to save per-repertoire results

        Returns:
            DataFrame with per-repertoire precision@k and recall@k columns
            for each k in ``ks``.
        """
        ks = sorted(set(int(k) for k in ks))
        max_k = max(ks)
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
            train_data = metadata[~test_mask]
            test_data = metadata[test_mask]

            train_files = train_data['file_path'].tolist()
            train_labels = train_data['label'].tolist()

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            # --- Train model (internal tuning handled by model.train()) ---
            model = self._make_model()
            train_result = model.train(train_files, train_labels)
            print(f"  Best C (k-mer): {train_result['best_c_kmer']}, "
                  f"Best C (V/J): {train_result['best_c_vj']}, "
                  f"Alpha: {train_result['best_alpha']:.2f}, "
                  f"Val AUROC: {train_result['val_auroc']:.4f}")

            # --- Score test repertoires ---
            print(f"\n--- Scoring test repertoires (ks={ks}) ---")
            fold_metrics = {k: {'precisions': [], 'recalls': []} for k in ks}

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Scoring"):
                file_path = row['file_path']
                filename_stem = os.path.basename(file_path) \
                    .replace('.tsv.gz', '').replace('.tsv', '')

                if filename_stem not in drivers_by_file:
                    continue

                driver_cdr3s = drivers_by_file[filename_stem]

                # Rank CDR3s by score (descending) once
                ranked = self.score_repertoire_cdr3s(file_path, model)

                row_result = {
                    'fold': test_fold,
                    'filename': filename_stem,
                    'participant_label': row[participant_col],
                    'disease_label': int(row['label']),
                    'n_repertoire_unique_cdr3s': len(ranked),
                    'n_ground_truth_drivers': len(driver_cdr3s),
                }

                for k in ks:
                    top_k_cdr3s = set(cdr3 for cdr3, _ in ranked[:k])
                    hits = top_k_cdr3s & driver_cdr3s
                    precision = len(hits) / k
                    recall = len(hits) / len(driver_cdr3s)

                    fold_metrics[k]['precisions'].append(precision)
                    fold_metrics[k]['recalls'].append(recall)

                    row_result[f'n_hits_at_{k}'] = len(hits)
                    row_result[f'precision_at_{k}'] = precision
                    row_result[f'recall_at_{k}'] = recall

                all_results.append(row_result)

            # --- Fold summary ---
            n_reps = len(fold_metrics[ks[0]]['precisions'])
            if n_reps > 0:
                print(f"\nFold {test_fold}: {n_reps} repertoires")
                fold_summary = {'fold': test_fold, 'n_repertoires': n_reps}
                for k in ks:
                    mean_prec = np.mean(fold_metrics[k]['precisions'])
                    mean_rec = np.mean(fold_metrics[k]['recalls'])
                    print(f"  Mean Precision@{k}: {mean_prec:.4f}")
                    print(f"  Mean Recall@{k}:    {mean_rec:.4f}")
                    fold_summary[f'mean_precision_at_{k}'] = mean_prec
                    fold_summary[f'mean_recall_at_{k}'] = mean_rec
                fold_summaries.append(fold_summary)
            else:
                print(f"\nFold {test_fold}: No test repertoires with "
                      f"ground truth drivers")
                fold_summary = {'fold': test_fold, 'n_repertoires': 0}
                for k in ks:
                    fold_summary[f'mean_precision_at_{k}'] = float('nan')
                    fold_summary[f'mean_recall_at_{k}'] = float('nan')
                fold_summaries.append(fold_summary)

            # Free memory between folds
            model.clear_cache()

        # ------------------------------------------------------------------
        # Overall results
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(all_results)

        print(f"\n{'=' * 60}")
        print(f"OVERALL RESULTS: {target_disease} Driver Identification "
              f"(ks={ks})")
        print(f"{'=' * 60}")
        print(f"Repertoires evaluated: {len(results_df)}")

        if len(results_df) > 0:
            total_possible = results_df['n_ground_truth_drivers'].sum()
            for k in ks:
                overall_precision = results_df[f'precision_at_{k}'].mean()
                overall_recall = results_df[f'recall_at_{k}'].mean()
                total_hits = results_df[f'n_hits_at_{k}'].sum()
                print(f"Overall Precision@{k} (macro): {overall_precision:.4f}")
                print(f"Overall Recall@{k}    (macro): {overall_recall:.4f}")
                if total_possible > 0:
                    micro_recall = total_hits / total_possible
                    micro_precision = total_hits / (len(results_df) * k)
                    print(f"Overall Precision@{k} (micro): {micro_precision:.4f}")
                    print(f"Overall Recall@{k}    (micro): {micro_recall:.4f}")

        print(f"\nPer-fold breakdown:")
        for s in fold_summaries:
            metrics_str = ", ".join(
                f"P@{k}={s[f'mean_precision_at_{k}']:.4f}, "
                f"R@{k}={s[f'mean_recall_at_{k}']:.4f}"
                for k in ks
            )
            print(f"  Fold {s['fold']}: {s['n_repertoires']} reps, "
                  f"{metrics_str}")

        if output_csv and len(results_df) > 0:
            results_df.to_csv(output_csv, index=False)
            print(f"\nPer-repertoire results saved to: {output_csv}")

        return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble Regression Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing repertoire .tsv.gz files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to evaluate (e.g. Covid19, HIV, Influenza)')
    parser.add_argument('--driver_seqs_path', type=str, required=True,
                        help='Path to ground truth driver sequences CSV')
    parser.add_argument('--k', type=int, nargs='+', required=True,
                        help='One or more top-ranked CDR3 cutoffs to evaluate '
                             '(e.g. --k 50 100 500)')
    parser.add_argument('--submodel', type=str, default='ensemble',
                        choices=['ensemble', 'kmer_only', 'vj_only'],
                        help='Sub-model to use (default: ensemble)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-repertoire results CSV')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Internal val fraction for alpha tuning (default: 0.2)')
    parser.add_argument('--n_cv_folds', type=int, default=5,
                        help='CV folds for C tuning (default: 5)')
    args = parser.parse_args()

    evaluator = EnsembleRegressionDriverIdentificationEvaluator(
        val_split=args.val_split,
        n_cv_folds=args.n_cv_folds,
        submodel=args.submodel,
    )

    results = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        ks=args.k,
        output_csv=args.output_csv,
    )
