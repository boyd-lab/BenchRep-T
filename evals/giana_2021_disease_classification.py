"""
Evaluation script for GIANA (Liu et al. 2021) disease classification.

Reference: Liu et al. 2021, "GIANA allows computationally-efficient TCR clustering
and multi-disease repertoire classification"

Approach
--------
For each CV fold, sequences from all training specimens (labelled 'train' with their
disease category) and all test specimens (labelled 'test' with their specimen ID) are
combined into a single GIANA input file.  GIANA clusters these sequences by sequence
similarity.  For each resulting cluster the disease fraction is computed from training
sequences only:

    disease_frac(cluster) = (#train seqs labelled as target disease)
                            / (#total train seqs in cluster)

Each test specimen receives a score equal to the mean disease fraction across all of
its sequences that appear in the cluster file.  Sequences that are unique and do not
cluster with anything are absent from GIANA output and contribute a score of 0.

This is methodologically equivalent to the paper's query mode (where reference clusters
are fixed from training data) but implemented as a single joint clustering step, which
is simpler to run and avoids GIANA's file-path assumptions in query mode.
"""

import os
import sys
import re
import argparse
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# GIANA source files live in models/GIANA/
_GIANA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'GIANA'
)

# ---------------------------------------------------------------------------
# Lazy GIANA module loader
# ---------------------------------------------------------------------------

_giana_module = None


def _load_giana_module():
    """
    Import GIANA4.1.py as a Python module (once per process).

    sys.path is extended so that GIANA4.1's `from query import *` and
    query.py's `from GIANA4 import *` resolve correctly.  GIANA4.py runs
    a one-time MDS fit on import; GIANA4.1 immediately overrides the result
    with its pre-computed hardcoded encoding matrix, so the cost is paid
    once per process and does not affect encoding correctness.
    """
    global _giana_module
    if _giana_module is not None:
        return _giana_module

    if _GIANA_DIR not in sys.path:
        sys.path.insert(0, _GIANA_DIR)

    spec = importlib.util.spec_from_file_location(
        'GIANA41', os.path.join(_GIANA_DIR, 'GIANA4.1.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _giana_module = mod
    return mod


def _build_vgene_scores(giana_mod):
    """
    Compute or load VgeneScores.txt and return the VScore dict for EncodeRepertoire.

    PreCalculateVgeneDist writes VgeneScores.txt to cwd, so we temporarily
    chdir to _GIANA_DIR where Imgt_Human_TRBV.fasta lives.
    """
    vgene_scores_path = os.path.join(_GIANA_DIR, 'VgeneScores.txt')
    if not os.path.exists(vgene_scores_path):
        old_cwd = os.getcwd()
        os.chdir(_GIANA_DIR)
        try:
            giana_mod.PreCalculateVgeneDist('Imgt_Human_TRBV.fasta')
        finally:
            os.chdir(old_cwd)

    VScore = {}
    with open(vgene_scores_path) as f:
        for line in f:
            ww = line.strip().split('\t')
            if len(ww) >= 3:
                VScore[(ww[0], ww[1])] = int(float(ww[2])) / 20
                VScore[(ww[1], ww[0])] = int(float(ww[2])) / 20
    return VScore

# Valid amino acid characters accepted by GIANA
_AA_PATTERN = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
# CDR3 lengths GIANA handles (from BuildLengthDict)
_GIANA_MIN_LEN = 10
_GIANA_MAX_LEN = 24


class GIANAEvaluator:
    """
    Evaluator for GIANA on binary disease classification.

    Reads AIRR .tsv.gz files, extracts CDR3 beta sequences (+ optional V-gene),
    combines training and test sequences into one GIANA input file per CV fold,
    runs GIANA clustering, then computes per-specimen disease scores as the mean
    fraction of co-clustered training TCRs that belong to the target disease.
    """

    HEALTHY_LABEL = "Healthy/Background"
    _GIANA_HEALTHY = "Healthy"

    def __init__(self,
                 sequence_col='cdr3_aa',
                 count_col='duplicate_count',
                 v_gene_col='v_call',
                 use_v_gene=True,
                 exact=False,
                 threshold_score=3.3,
                 threshold_iso=5,
                 threshold_vgene=3.7,
                 n_threads=1,
                 use_gpu=False,
                 max_seqs_per_specimen=None,
                 indices_map=None,
                 results_dir='results/giana',
                 debug=False,
                 debug_repertoires=10):
        """
        Args:
            sequence_col: AIRR column with CDR3 amino acid sequences.
            count_col: AIRR column with duplicate counts; uses 1 if absent.
            v_gene_col: AIRR column with V-gene calls; pass None to omit V-gene.
            use_v_gene: If False, omit V-gene features in GIANA clustering.
            exact: If True, run Smith-Waterman alignment after FAISS (10x slower;
                   not recommended for >1M sequences). Default False.
            threshold_score: Smith-Waterman score threshold (default 3.3, as in paper;
                             only used when exact=True).
            threshold_iso: Isometric distance threshold. Default 5 for non-exact mode
                           (recommended by authors); use 7 for exact mode.
            threshold_vgene: V-gene similarity threshold (default 3.7).
            n_threads: Number of FAISS CPU threads.
            use_gpu: If True, use GPU-accelerated FAISS index (requires faiss-gpu).
            max_seqs_per_specimen: If set, cap sequences per specimen (top by count).
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
                         rep_id is the filename without extension, e.g.
                         'part_table_PARTICIPANT_SPECIMEN'.
            results_dir: Base directory for GIANA cluster output files.
            debug: If True, load only debug_repertoires specimens per class.
            debug_repertoires: Number of repertoires per class in debug mode.
        """
        self.sequence_col = sequence_col
        self.count_col = count_col
        self.v_gene_col = v_gene_col
        self.use_v_gene = use_v_gene and (v_gene_col is not None)
        self.exact = exact
        self.threshold_score = threshold_score
        self.threshold_iso = threshold_iso
        self.threshold_vgene = threshold_vgene
        self.n_threads = n_threads
        self.use_gpu = use_gpu
        self.max_seqs_per_specimen = max_seqs_per_specimen
        self.indices_map = indices_map
        self.results_dir = results_dir
        self.debug = debug
        self.debug_repertoires = debug_repertoires

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
    # Sequence loading and GIANA input preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_v_gene(v_gene_str):
        """Normalize V-gene call to IMGT format (e.g., TRBV10-3*01)."""
        if pd.isna(v_gene_str) or str(v_gene_str).strip() == '':
            return None
        v = str(v_gene_str).strip().split(',')[0].strip()
        if not v:
            return None
        if '*' not in v:
            v = v + '*01'
        return v

    def _load_repertoire(self, file_path):
        """
        Load a repertoire file and return a cleaned DataFrame.

        Returns DataFrame with columns: [sequence_col, (v_gene_col), count_col],
        deduplicated by CDR3 (+ V gene), sorted by count descending.
        Returns None if the file is unreadable or yields no valid sequences.
        """
        wanted_cols = {self.sequence_col, self.count_col}
        if self.use_v_gene:
            wanted_cols.add(self.v_gene_col)

        try:
            df = pd.read_csv(file_path, sep='\t',
                             usecols=lambda c: c in wanted_cols)
            if self.indices_map is not None:
                rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
                indices = self.indices_map.get(rep_id)
                if indices is not None:
                    df = df.iloc[indices]
        except Exception as e:
            print(f"  Warning: could not read {file_path}: {e}")
            return None

        if self.sequence_col not in df.columns:
            print(f"  Warning: '{self.sequence_col}' not in {file_path}, skipping.")
            return None

        if self.count_col not in df.columns:
            df[self.count_col] = 1
        df[self.count_col] = pd.to_numeric(df[self.count_col], errors='coerce').fillna(1)
        df = df[df[self.count_col] >= 1].copy()

        # Filter to valid amino acid sequences in GIANA's length range
        df = df.dropna(subset=[self.sequence_col])
        df = df[df[self.sequence_col].str.match(_AA_PATTERN, na=False)]
        df = df[df[self.sequence_col].str.len().between(_GIANA_MIN_LEN, _GIANA_MAX_LEN)]

        if len(df) == 0:
            return None

        # Normalize and filter V gene if used
        if self.use_v_gene and self.v_gene_col in df.columns:
            df[self.v_gene_col] = df[self.v_gene_col].apply(self._normalize_v_gene)
            df = df.dropna(subset=[self.v_gene_col])
            group_cols = [self.sequence_col, self.v_gene_col]
        else:
            group_cols = [self.sequence_col]

        if len(df) == 0:
            return None

        # Deduplicate and sort by count
        df = (
            df.groupby(group_cols, as_index=False)[self.count_col]
            .sum()
            .sort_values(self.count_col, ascending=False)
            .reset_index(drop=True)
        )

        if self.max_seqs_per_specimen is not None:
            df = df.head(self.max_seqs_per_specimen)

        return df

    def _build_giana_rows(self, df, source_tag, label):
        """
        Convert a repertoire DataFrame to GIANA input rows.

        GIANA input format (no header, tab-separated):
          col0: CDR3 aa sequence
          col1: V gene (IMGT format; placeholder 'TRBV2*01' when not used)
          col2: source_tag  ('train' or 'test')
          col3: label       (disease label for train, specimen_label for test)

        After GIANA clustering inserts a cluster_id column after col0, the
        cluster file columns become: CDR3, cluster_id, V_gene, source_tag, label.
        """
        _PLACEHOLDER_V = 'TRBV2*01'
        rows = []
        for _, row in df.iterrows():
            cdr3 = row[self.sequence_col]
            if self.use_v_gene and self.v_gene_col in df.columns:
                v = row.get(self.v_gene_col, _PLACEHOLDER_V) or _PLACEHOLDER_V
            else:
                v = _PLACEHOLDER_V
            rows.append(f"{cdr3}\t{v}\t{source_tag}\t{label}")
        return rows

    def _write_giana_input(self, lines, outpath):
        """Write GIANA input file (no header)."""
        with open(outpath, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def _run_giana(self, input_file, work_dir):
        """
        Run GIANA clustering on input_file by calling EncodeRepertoire directly.

        Returns path to the produced cluster file.
        """
        basename = os.path.basename(input_file)
        # GIANA removes the last file extension (matching [txcsv]+) then appends
        # '--RotationEncodingBL62.txt'
        base_no_ext = re.sub(r'\.[txcsv]+$', '', basename)
        outfile = os.path.join(work_dir, base_no_ext + '--RotationEncodingBL62.txt')

        giana = _load_giana_module()

        # Set FAISS thread count
        import faiss
        faiss.omp_set_num_threads(self.n_threads)

        # Build V-gene score dict (reads/writes VgeneScores.txt once)
        VScore = _build_vgene_scores(giana) if self.use_v_gene else {}

        n_lines = sum(1 for _ in open(input_file))
        print(f"  Running GIANA on {basename} ({n_lines:,} sequences) "
              f"[exact={self.exact}, gpu={self.use_gpu}] ...")
        giana.EncodeRepertoire(
            input_file, work_dir, '',
            exact=self.exact, ST=3,
            thr_v=self.threshold_vgene,
            thr_s=self.threshold_score,
            VDict=VScore,
            Vgene=self.use_v_gene,
            thr_iso=self.threshold_iso,
            gap=-6, GPU=self.use_gpu, Mat=False, verbose=True,
        )

        if not os.path.exists(outfile):
            raise RuntimeError(
                f"GIANA did not produce expected output file: {outfile}"
            )
        return outfile

    def _parse_cluster_file(self, cluster_file):
        """
        Parse GIANA cluster output (skip 2 header lines).

        Expected columns after joint clustering of our combined input:
          0: CDR3
          1: cluster_id
          2: V_gene
          3: source_tag  ('train' or 'test')
          4: label       (disease label or specimen_label)
        """
        df = pd.read_csv(cluster_file, sep='\t', header=None, skiprows=2)
        return df

    def _compute_specimen_scores(self, cluster_df, target_disease):
        """
        Compute per-specimen disease scores from the cluster DataFrame.

        For each cluster: disease_frac = (#train rows with label==target_disease)
                                         / (#total train rows).
        For each test specimen: score = mean disease_frac over all its sequences.
        Sequences absent from the cluster file (unclustered singletons) contribute 0.

        Returns:
            dict mapping specimen_label -> float score in [0, 1].
        """
        if cluster_df.shape[1] < 5:
            print("Warning: cluster file has fewer columns than expected; "
                  "column layout may be wrong.")
            return {}

        train_mask = cluster_df[3] == 'train'
        test_mask = cluster_df[3] == 'test'
        train_df = cluster_df[train_mask]
        test_df = cluster_df[test_mask]

        if len(test_df) == 0:
            return {}

        # Disease fraction per cluster (from training sequences only)
        cluster_disease_frac = {}
        for cluster_id, grp in train_df.groupby(1):
            n_disease = (grp[4] == target_disease).sum()
            cluster_disease_frac[cluster_id] = n_disease / max(len(grp), 1)

        # Aggregate per test specimen
        specimen_fracs: dict[str, list[float]] = {}
        for _, row in test_df.iterrows():
            specimen = row[4]
            frac = cluster_disease_frac.get(row[1], 0.0)
            specimen_fracs.setdefault(specimen, []).append(frac)

        return {s: float(np.mean(fracs)) for s, fracs in specimen_fracs.items()}

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def run_cross_validation(self, metadata_path, target_disease, data_dir,
                              participant_col='participant_label',
                              file_prefix='part_table_', file_suffix='.tsv.gz',
                              disease_col='disease',
                              fold_col='malid_cross_validation_fold_id_when_in_test_set',
                              n_folds=3,
                              random_state=None,
                              tune_parameters=True,
                              p_value_candidates=None,
                              allowed_participants=None):
        """
        Run k-fold cross-validation using pre-defined fold assignments.

        For each fold, a joint GIANA input file is built from all training
        sequences (labelled 'train') and all test sequences (labelled 'test'),
        GIANA is run once per fold, and per-specimen disease scores are derived
        from the cluster output.

        Args:
            metadata_path: Path to metadata.tsv.
            target_disease: Disease to classify against Healthy/Background.
            data_dir: Directory containing AIRR .tsv.gz files.
            participant_col: Column with participant labels.
            file_prefix: Filename prefix (default 'part_table_').
            file_suffix: Filename suffix (default '.tsv.gz').
            disease_col: Column with disease labels.
            fold_col: Column with pre-defined test-fold IDs (0, 1, 2, …).
            n_folds: Number of cross-validation folds.
            random_state: Accepted for API compatibility; unused (GIANA is unsupervised).
            tune_parameters: Accepted for API compatibility; unused.
            p_value_candidates: Accepted for API compatibility; unused.
            allowed_participants: Optional set of specimen_labels to restrict to.

        Returns:
            pd.DataFrame with per-sample scores across all folds.
        """
        os.makedirs(self.results_dir, exist_ok=True)

        raw_metadata = self.load_metadata(metadata_path)
        metadata = self.prepare_disease_data(raw_metadata, target_disease, disease_col)
        metadata = self.add_file_paths(metadata, data_dir, participant_col,
                                       file_prefix, file_suffix)
        metadata = self.filter_existing_files(metadata)

        if self.debug:
            disease_rows = metadata[metadata['label'] == 1].head(self.debug_repertoires)
            healthy_rows = metadata[metadata['label'] == 0].head(self.debug_repertoires)
            metadata = pd.concat([disease_rows, healthy_rows], ignore_index=True)
            print(f"[DEBUG] Restricted to {len(metadata)} repertoires "
                  f"({len(disease_rows)} disease, {len(healthy_rows)} healthy).")

        if allowed_participants is not None:
            before = len(metadata)
            metadata = metadata[metadata['specimen_label'].isin(allowed_participants)]
            print(f"Filtered to {len(metadata)} of {before} specimens "
                  f"based on allowed_participants.")

        all_test_rows = []
        all_probs = []
        all_labels = []
        fold_results = []

        for test_fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {test_fold}: Test fold = {test_fold}")
            print(f"{'='*60}")

            test_data = metadata[metadata[fold_col] == test_fold]
            train_data = metadata[metadata[fold_col] != test_fold]

            print(f"Train: {len(train_data)}, Test: {len(test_data)}")

            fold_dir = os.path.join(
                self.results_dir, f"{target_disease}_fold{test_fold}"
            )
            os.makedirs(fold_dir, exist_ok=True)

            combined_file = os.path.join(
                fold_dir, f"combined_fold{test_fold}.txt"
            )

            # Build combined GIANA input file
            print("\nLoading training sequences...")
            giana_lines = []
            n_train_loaded = 0

            for _, row in train_data.iterrows():
                disease_label = (
                    target_disease if row['label'] == 1 else self._GIANA_HEALTHY
                )
                df = self._load_repertoire(row['file_path'])
                if df is None or len(df) == 0:
                    continue
                giana_lines.extend(
                    self._build_giana_rows(df, 'train', disease_label)
                )
                n_train_loaded += 1

            print(f"  Loaded {len(giana_lines):,} train sequences "
                  f"from {n_train_loaded} specimens.")

            print("Loading test sequences...")
            n_test_loaded = 0
            for _, row in test_data.iterrows():
                df = self._load_repertoire(row['file_path'])
                if df is None or len(df) == 0:
                    continue
                giana_lines.extend(
                    self._build_giana_rows(df, 'test', row['specimen_label'])
                )
                n_test_loaded += 1

            print(f"  Loaded test sequences from {n_test_loaded} specimens.")
            print(f"Total sequences for GIANA: {len(giana_lines):,}")

            self._write_giana_input(giana_lines, combined_file)

            # Run GIANA clustering
            cluster_file = self._run_giana(combined_file, fold_dir)

            # Parse cluster output and compute scores
            print("Parsing cluster output...")
            cluster_df = self._parse_cluster_file(cluster_file)
            scores = self._compute_specimen_scores(cluster_df, target_disease)

            print(f"Scored {len(scores)} test specimens.")

            # Collect per-specimen results
            fold_probs = []
            fold_labels = []

            for _, row in test_data.iterrows():
                specimen = row['specimen_label']
                if specimen not in scores:
                    print(f"  Warning: no cluster-file entries for specimen '{specimen}'; "
                          f"assigning score 0.")
                    score = 0.0
                else:
                    score = scores[specimen]

                true_label = int(row['label'])
                fold_probs.append(score)
                fold_labels.append(true_label)
                all_test_rows.append({
                    'participant_label': row[participant_col],
                    'specimen_label': specimen,
                    'disease_label': true_label,
                    'disease_label_str': row[disease_col],
                    'method': 'GIANA',
                    'disease_model': target_disease,
                    'model_score': score,
                    'malid_cross_validation_fold_id_when_in_test_set': test_fold,
                })

            if len(fold_labels) < 2 or len(set(fold_labels)) < 2:
                print(f"Warning: fold {test_fold} has <2 classes in test; "
                      f"skipping fold metrics.")
                all_probs.extend(fold_probs)
                all_labels.extend(fold_labels)
                continue

            fold_auroc = roc_auc_score(fold_labels, fold_probs)
            fold_aupr = average_precision_score(fold_labels, fold_probs)
            print(f"Test AUROC: {fold_auroc:.4f}, Test AUPR: {fold_aupr:.4f}")

            fold_results.append({
                'fold': test_fold,
                'test_auroc': fold_auroc,
                'test_aupr': fold_aupr,
            })
            all_probs.extend(fold_probs)
            all_labels.extend(fold_labels)

        if len(all_labels) >= 2 and len(set(all_labels)) >= 2:
            all_probs_arr = np.array(all_probs)
            all_labels_arr = np.array(all_labels)
            overall_auroc = roc_auc_score(all_labels_arr, all_probs_arr)
            overall_aupr = average_precision_score(all_labels_arr, all_probs_arr)

            print(f"\n{'='*60}")
            print(f"OVERALL RESULTS: {target_disease} vs Healthy")
            print(f"{'='*60}")
            if fold_results:
                fold_aurocs = [r['test_auroc'] for r in fold_results]
                fold_auprs = [r['test_aupr'] for r in fold_results]
                print(f"Mean Test AUROC: {np.mean(fold_aurocs):.4f} "
                      f"± {np.std(fold_aurocs):.4f}")
                print(f"Mean Test AUPR:  {np.mean(fold_auprs):.4f}  "
                      f"± {np.std(fold_auprs):.4f}")
            print(f"Overall AUROC (all folds combined): {overall_auroc:.4f}")
            print(f"Overall AUPR  (all folds combined): {overall_aupr:.4f}")

        return pd.DataFrame(all_test_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="GIANA Disease Classification Evaluation"
    )
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Directory containing AIRR .tsv.gz repertoire files')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Disease to classify (e.g. Lupus, T1D, HIV, Covid19)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save per-sample scores CSV (optional)')
    parser.add_argument('--results_dir', type=str, default='results/giana',
                        help='Directory for GIANA cluster files (default: results/giana)')
    parser.add_argument('--exact', action='store_true',
                        help='Enable Smith-Waterman exact mode (10x slower; not recommended '
                             'for >1M sequences). Default: non-exact mode.')
    parser.add_argument('--threshold_score', type=float, default=3.3,
                        help='Smith-Waterman score threshold; only used with --exact (default: 3.3)')
    parser.add_argument('--threshold_iso', type=float, default=5,
                        help='Isometric distance threshold (default: 5 for non-exact mode)')
    parser.add_argument('--threshold_vgene', type=float, default=3.7,
                        help='V-gene similarity threshold (default: 3.7)')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of FAISS CPU threads (default: 1)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU-accelerated FAISS index (requires faiss-gpu)')
    parser.add_argument('--no_v_gene', action='store_true',
                        help='Disable V-gene features (CDR3 only)')
    parser.add_argument('--max_seqs_per_specimen', type=int, default=None,
                        help='Cap sequences per specimen (top by count; default: no cap)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: load only a small number of repertoires per class')
    parser.add_argument('--debug_repertoires', type=int, default=10,
                        help='Repertoires per class to load in debug mode (default: 10)')
    args = parser.parse_args()

    evaluator = GIANAEvaluator(
        use_v_gene=not args.no_v_gene,
        exact=args.exact,
        threshold_score=args.threshold_score,
        threshold_iso=args.threshold_iso,
        threshold_vgene=args.threshold_vgene,
        n_threads=args.n_threads,
        use_gpu=args.use_gpu,
        max_seqs_per_specimen=args.max_seqs_per_specimen,
        results_dir=args.results_dir,
        debug=args.debug,
        debug_repertoires=args.debug_repertoires,
    )

    scores_df = evaluator.run_cross_validation(
        metadata_path=args.metadata_path,
        target_disease=args.target_disease,
        data_dir=args.repertoire_data_dir,
    )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
