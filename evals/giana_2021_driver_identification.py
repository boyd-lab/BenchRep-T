"""
Driver sequence identification evaluation for GIANA (Liu et al. 2021).

Per-sequence importance is the disease fraction of the reference cluster to
which each test CDR3 was assigned.  Higher disease fraction indicates that
most reference TCRs sharing the same structural neighbourhood came from
disease samples, which GIANA uses as evidence of disease association.

Reference clusters whose membership exceeds 100 TCRs are excluded (per the
paper), and test TCRs that have no close reference neighbour receive score 0.

Multiple rows sharing the same CDR3 (but differing V genes) can be assigned
to different clusters.  Scores are aggregated by taking the maximum disease
fraction across all clusters that contain that CDR3.

Sequences are ranked descending by cluster disease fraction and the top-k
CDR3s are compared against ground truth driver sequences.

Metrics: precision@k and recall@k, macro-averaged across repertoires.
"""

import os
import sys
import re
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.giana_2021_disease_classification import (
    GIANAEvaluator,
    _load_giana_module,
    _build_vgene_scores,
    _AA_PATTERN,
    _GIANA_MIN_LEN,
    _GIANA_MAX_LEN,
)


class GIANADriverIdentificationEvaluator(GIANAEvaluator):
    """
    Evaluates GIANA's ability to identify disease-associated driver sequences
    using per-CDR3 cluster disease fractions.
    """

    # ------------------------------------------------------------------
    # Per-CDR3 scoring from merged file
    # ------------------------------------------------------------------

    def score_test_cdr3s(self, merged_file, target_disease):
        """
        Extract per-CDR3 disease fraction scores from the MergeExist output.

        Disease fractions come from reference rows (role=='ref').  Clusters
        with >100 reference members are excluded.  Each test CDR3 receives
        the disease fraction of its assigned cluster (0.0 if unassigned or
        in an excluded cluster).  When a CDR3 appears in multiple clusters
        (e.g. paired with different V genes), the maximum fraction is kept.

        Args:
            merged_file: Path to MergeExist output (6-column TSV, no header).
            target_disease: Disease label string to score against.

        Returns:
            dict mapping specimen_label → list of (cdr3, score) sorted desc.
        """
        try:
            df = pd.read_csv(merged_file, sep='\t', header=None)
        except Exception as e:
            print(f"Warning: could not read merged file {merged_file}: {e}")
            return {}

        if df.shape[1] < 6:
            print("Warning: merged file has fewer than 6 columns.")
            return {}

        ref_df = df[df[5] == 'ref']
        query_df = df[(df[5] == 'query') & (df[3] == 'test')]

        if len(query_df) == 0:
            return {}

        # Exclude clusters whose reference membership exceeds 100 TCRs.
        cluster_ref_size = ref_df.groupby(1).size()
        excluded_clusters = {
            cid for cid, sz in cluster_ref_size.items() if sz > 100
        }

        # Disease fraction per reference cluster.
        cluster_disease_frac = {}
        for cluster_id, grp in ref_df.groupby(1):
            if cluster_id in excluded_clusters:
                continue
            n_disease = (grp[4] == target_disease).sum()
            cluster_disease_frac[cluster_id] = n_disease / max(len(grp), 1)

        # Aggregate per CDR3 per specimen: max disease fraction across clusters.
        specimen_cdr3_scores: dict[str, dict[str, float]] = {}
        for _, row in query_df.iterrows():
            cluster_id = row[1]
            cdr3 = row[0]
            specimen = row[4]

            if cluster_id in excluded_clusters:
                score = 0.0
            else:
                score = cluster_disease_frac.get(cluster_id, 0.0)

            cdr3_map = specimen_cdr3_scores.setdefault(specimen, {})
            cdr3_map[cdr3] = max(cdr3_map.get(cdr3, 0.0), score)

        # Convert to sorted lists.
        return {
            specimen: sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for specimen, scores in specimen_cdr3_scores.items()
        }

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
    # Main cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation_driver_id(
        self, metadata_path, target_disease, data_dir,
        driver_seqs_path, k,
        participant_col='participant_label',
        file_prefix='part_table_', file_suffix='.tsv.gz',
        disease_col='disease',
        fold_col='malid_cross_validation_fold_id_when_in_test_set',
        n_folds=3,
        allowed_participants=None,
        cluster_dir=None,
        output_csv=None,
    ):
        """
        Run k-fold CV for driver sequence identification using GIANA.

        For each fold:
        1. Build reference clusters from training sequences.
        2. Assign test sequences to reference clusters via GIANA query mode.
        3. Score each test CDR3 by its cluster's disease fraction.
        4. Compare top-k CDR3s against ground truth driver sequences.

        Returns:
            DataFrame with per-repertoire precision@k and recall@k.
        """
        os.makedirs(self.results_dir, exist_ok=True)

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

            test_data = metadata[metadata[fold_col] == test_fold]
            train_data = metadata[metadata[fold_col] != test_fold]

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            fold_dir = os.path.join(
                self.results_dir, f"{target_disease}_driver_fold{test_fold}"
            )
            os.makedirs(fold_dir, exist_ok=True)

            # ----------------------------------------------------------
            # Try to reuse cluster files from disease classification run
            # ----------------------------------------------------------
            reused_merged = None
            if cluster_dir is not None:
                candidate = os.path.join(
                    cluster_dir, f"{target_disease}_fold{test_fold}", "queryFinal.txt"
                )
                if os.path.isfile(candidate):
                    print(f"  Reusing cluster file from disease classification: {candidate}")
                    reused_merged = candidate

            if reused_merged is not None:
                merged_file = reused_merged
            else:
                # ----------------------------------------------------------
                # Phase 1: Build reference clusters from training data
                # ----------------------------------------------------------
                print("\nLoading training sequences...")
                train_lines = []
                n_train_loaded = 0

                for _, row in train_data.iterrows():
                    disease_label = (
                        target_disease if row['label'] == 1 else self._GIANA_HEALTHY
                    )
                    df = self._load_repertoire(row['file_path'])
                    if df is None or len(df) == 0:
                        continue
                    train_lines.extend(
                        self._build_giana_rows(df, 'train', disease_label)
                    )
                    n_train_loaded += 1

                print(f"  Loaded {len(train_lines):,} train sequences "
                      f"from {n_train_loaded} specimens.")

                train_file = os.path.join(fold_dir, f"train_fold{test_fold}.txt")
                self._write_giana_input(train_lines, train_file)

                print("Clustering training sequences to build reference...")
                ref_cluster_file = self._run_giana(train_file, fold_dir)

                giana = _load_giana_module()
                print("Encoding reference sequences for query mode...")
                rData = giana.CreateReference(train_file, Vgene=self.use_v_gene, ST=3)

                # ----------------------------------------------------------
                # Phase 2: Assign test sequences to reference clusters
                # ----------------------------------------------------------
                print("\nLoading test sequences...")
                test_lines = []
                n_test_loaded = 0

                for _, row in test_data.iterrows():
                    df = self._load_repertoire(row['file_path'])
                    if df is None or len(df) == 0:
                        continue
                    test_lines.extend(
                        self._build_giana_rows(df, 'test', row['specimen_label'])
                    )
                    n_test_loaded += 1

                print(f"  Loaded test sequences from {n_test_loaded} specimens.")

                query_file = os.path.join(fold_dir, f"query_fold{test_fold}.txt")
                self._write_giana_input(test_lines, query_file)

                merged_file = self._run_query(
                    query_file, rData, ref_cluster_file, fold_dir
                )

            # ----------------------------------------------------------
            # Per-CDR3 scoring and evaluation
            # ----------------------------------------------------------
            if merged_file is None:
                cdr3_scores_by_specimen = {}
            else:
                print("Extracting per-CDR3 disease fraction scores...")
                cdr3_scores_by_specimen = self.score_test_cdr3s(
                    merged_file, target_disease
                )

            print(f"\n--- Evaluating driver identification (k={k}) ---")
            fold_precisions = []
            fold_recalls = []

            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                               desc="Evaluating"):
                file_path = row['file_path']
                filename_stem = (
                    os.path.basename(file_path)
                    .replace('.tsv.gz', '').replace('.tsv', '')
                )
                specimen = row['specimen_label']

                if filename_stem not in drivers_by_file:
                    continue

                driver_cdr3s = drivers_by_file[filename_stem]
                ranked = cdr3_scores_by_specimen.get(specimen, [])
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
        description="GIANA Driver Sequence Identification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--repertoire_data_dir', type=str, required=True)
    parser.add_argument('--target_disease', type=str, required=True)
    parser.add_argument('--driver_seqs_path', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results/giana_driver')
    parser.add_argument('--cluster_dir', type=str, default=None,
                        help='results_dir from giana_2021_disease_classification.py run; '
                             'driver eval reuses its queryFinal.txt files instead of rebuilding.')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--threshold_score', type=float, default=3.3)
    parser.add_argument('--threshold_iso', type=float, default=5)
    parser.add_argument('--threshold_vgene', type=float, default=3.7)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--no_v_gene', action='store_true')
    parser.add_argument('--max_seqs_per_specimen', type=int, default=None)
    args = parser.parse_args()

    evaluator = GIANADriverIdentificationEvaluator(
        use_v_gene=not args.no_v_gene,
        exact=args.exact,
        threshold_score=args.threshold_score,
        threshold_iso=args.threshold_iso,
        threshold_vgene=args.threshold_vgene,
        n_threads=args.n_threads,
        use_gpu=args.use_gpu,
        max_seqs_per_specimen=args.max_seqs_per_specimen,
        results_dir=args.results_dir,
    )

    results = evaluator.run_cross_validation_driver_id(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
        driver_seqs_path=args.driver_seqs_path,
        k=args.k,
        cluster_dir=args.cluster_dir,
        output_csv=args.output_csv,
    )
