"""
GIANA 2020 TCR Disease Classification Model.

Implementation of a disease classifier based on GIANA (Geometric Isometry based
ANtigen-specific tcr Alignment) from:
Zhang et al., "GIANA allows computationally-efficient TCR clustering and
multi-disease repertoire classification by isometric transformation" (2020)

This module uses GIANA's isometric CDR3 encoding and FAISS nearest-neighbor
clustering to identify disease-enriched TCR clusters, then applies Fisher's
exact test for cluster selection and Beta-Binomial modeling for classification.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from scipy.special import betaln
from scipy.optimize import minimize
from tqdm import tqdm
import faiss


def _load_giana(giana_dir):
    """
    Import GIANA encoding functions from the GIANA installation directory.

    Returns a dict with the key objects: EncodingCDR3, bl62np, M6, n0, Ndim.
    """
    if giana_dir not in sys.path:
        sys.path.insert(0, giana_dir)
    import GIANA4
    return {
        'EncodingCDR3': GIANA4.EncodingCDR3,
        'bl62np': GIANA4.bl62np,
        'M6': GIANA4.M6,
        'n0': GIANA4.n0,
        'Ndim': GIANA4.Ndim,
    }


# Module-level cache so GIANA is only imported once
_giana_cache = {}


def _get_giana(giana_dir):
    """Get or load GIANA functions (cached)."""
    if giana_dir not in _giana_cache:
        _giana_cache[giana_dir] = _load_giana(giana_dir)
    return _giana_cache[giana_dir]


class GIANA_Classifier:
    """
    GIANA-based TCR disease classifier.

    Uses GIANA's isometric CDR3 encoding to cluster similar TCR sequences,
    identifies disease-enriched clusters via Fisher's exact test, and uses
    a Beta-Binomial generative model for classification.
    """

    # CDR3 length range supported by GIANA
    SUPPORTED_LENGTHS = list(range(10, 25))  # 10 to 24 inclusive

    # Terminal residues stripped per GIANA convention
    ST = 3   # Skip first 3 residues
    ED = 2   # Skip last 2 residues

    def __init__(self, iso_threshold=7, p_value_threshold=1e-4,
                 sequence_col='cdr3_aa',
                 giana_dir='/users/chihoim/software/GIANA',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None):
        """
        Initialize the GIANA classifier.

        Args:
            iso_threshold: Isometric distance threshold for FAISS clustering (default: 7)
            p_value_threshold: P-value cutoff for Fisher's exact test on clusters (default: 1e-4)
            sequence_col: Column name containing CDR3 sequences (default: 'cdr3_aa')
            giana_dir: Path to GIANA installation directory
            subsample_fraction: Fraction of reads to keep for depth simulation (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling (default: 42)
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set)
        """
        self.iso_threshold = iso_threshold
        self.p_value_threshold = p_value_threshold
        self.sequence_col = sequence_col
        self.giana_dir = giana_dir
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n

        # Load GIANA functions
        giana = _get_giana(giana_dir)
        self._encode_fn = giana['EncodingCDR3']
        self._bl62np = giana['bl62np']
        self._M6 = giana['M6']
        self._n0 = giana['n0']
        self._ndim = giana['Ndim']

        # Valid amino acid set (from GIANA's encoding dictionary)
        self._valid_aas = set(self._bl62np.keys())

        # Caches
        self._repertoire_cache = {}
        self._cluster_stats_cache = None

        # Trained model state
        self.diagnostic_clusters = set()
        self._faiss_indices = {}        # {length -> faiss.IndexFlatL2}
        self._cluster_assignments = {}  # {length -> np.array of cluster IDs}
        self.model_params = {}          # Beta-Binomial alpha/beta per class

    def load_repertoire(self, file_path, use_cache=True):
        """
        Load a single patient's repertoire from a TSV file.

        Args:
            file_path: Path to the repertoire TSV file (supports .tsv and .tsv.gz)
            use_cache: Whether to use cached data if available (default: True)

        Returns:
            Set of unique CDR3 sequences
        """
        if use_cache and file_path in self._repertoire_cache:
            return self._repertoire_cache[file_path]

        try:
            df = pd.read_csv(file_path, sep='\t')
            if self.sequence_col not in df.columns:
                raise ValueError(f"Column '{self.sequence_col}' not found in {file_path}")
            # Subsample rows to simulate reduced sequencing depth
            if self.subsample_n is not None:
                df = df.sample(n=min(self.subsample_n, len(df)), random_state=self.subsample_seed)
            elif self.subsample_fraction < 1.0:
                df = df.sample(frac=self.subsample_fraction, random_state=self.subsample_seed)
            sequences = set(df[self.sequence_col].dropna().unique())

            if use_cache:
                self._repertoire_cache[file_path] = sequences
            return sequences
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return set()

    def preload_repertoires(self, file_paths):
        """
        Preload all repertoire files into cache.

        Args:
            file_paths: List of file paths to preload
        """
        print(f"Preloading {len(file_paths)} repertoire files...")
        for file_path in tqdm(file_paths, desc="Preloading repertoires"):
            self.load_repertoire(file_path, use_cache=True)
        print(f"Cached {len(self._repertoire_cache)} repertoires.")

    def clear_cache(self):
        """Clear all caches to free memory."""
        self._repertoire_cache = {}
        self._cluster_stats_cache = None
        self._faiss_indices = {}
        self._cluster_assignments = {}

    def _get_effective_threshold(self, length):
        """
        Adjust isometric distance threshold by CDR3 length.

        GIANA uses shorter thresholds for shorter CDR3s.
        From EncodeRepertoire line 747: thr_iso - 0.5*(15-length)
        """
        return self.iso_threshold - 0.5 * (15 - length)

    def _encode_cdr3(self, cdr3):
        """
        Encode a single CDR3 using GIANA's isometric encoding.

        Strips first 3 and last 2 residues, then applies rotational encoding.

        Args:
            cdr3: CDR3 amino acid sequence string

        Returns:
            96-D float32 numpy vector, or None if CDR3 is invalid
        """
        if len(cdr3) < self.ST + self.ED + 1:
            return None
        core = cdr3[self.ST:-self.ED]
        if any(aa not in self._valid_aas for aa in core):
            return None
        vec = self._encode_fn(core, self._M6, self._n0)
        return vec.astype('float32')

    def _encode_batch(self, cdr3_list):
        """
        Encode a list of CDR3 sequences. Returns (encoded_matrix, valid_indices).

        Args:
            cdr3_list: List of CDR3 strings (all same length)

        Returns:
            Tuple of (N x 96 float32 matrix, list of valid indices into cdr3_list)
        """
        encoded = []
        valid_idx = []
        for i, cdr3 in enumerate(cdr3_list):
            vec = self._encode_cdr3(cdr3)
            if vec is not None:
                encoded.append(vec)
                valid_idx.append(i)
        if not encoded:
            return np.empty((0, self._ndim * 6), dtype='float32'), []
        return np.array(encoded, dtype='float32'), valid_idx

    def compute_cluster_statistics(self, training_files, training_labels):
        """
        Pool all training CDR3s, encode with GIANA, cluster per length group,
        and compute Fisher's exact test p-values per cluster.

        This is expensive but only runs once per fold (independent of p_value_threshold).

        Args:
            training_files: List of file paths to patient .tsv files
            training_labels: List of integers (1 for Diseased, 0 for Healthy)

        Returns:
            Dictionary with cluster statistics including p-values
        """
        print("--- Computing GIANA Cluster Statistics (one-time) ---")

        total_pos = sum(training_labels)
        total_neg = len(training_labels) - total_pos
        print(f"Analyzing {len(training_files)} subjects ({total_pos} Pos, {total_neg} Neg)...")

        # Step 1: Pool all unique CDR3s, tracking which subjects each appears in
        tcr_to_subjects = {}  # {cdr3_str -> set of subject indices}

        for idx, (file_path, label) in enumerate(tqdm(
                zip(training_files, training_labels),
                total=len(training_files), desc="Pooling CDR3s")):
            sequences = self.load_repertoire(file_path)
            for seq in sequences:
                if seq not in tcr_to_subjects:
                    tcr_to_subjects[seq] = set()
                tcr_to_subjects[seq].add(idx)

        print(f"Total unique CDR3s pooled: {len(tcr_to_subjects)}")

        # Step 2: Group by length, encode, cluster
        global_cluster_id = 0
        cluster_subject_sets = {}  # {cluster_id -> set of subject indices}

        self._faiss_indices = {}
        self._cluster_assignments = {}

        for length in self.SUPPORTED_LENGTHS:
            length_tcrs = [seq for seq in tcr_to_subjects if len(seq) == length]
            if not length_tcrs:
                continue

            # Encode
            encoded_matrix, valid_idx = self._encode_batch(length_tcrs)
            if len(valid_idx) == 0:
                continue

            valid_tcrs = [length_tcrs[i] for i in valid_idx]
            n_seqs = encoded_matrix.shape[0]

            # Cluster using FAISS range_search + union-find
            effective_thr = self._get_effective_threshold(length)
            dim = self._ndim * 6
            index = faiss.IndexFlatL2(dim)
            index.add(encoded_matrix)

            lims, D, I = index.range_search(encoded_matrix, effective_thr)

            # Union-find for connected components
            parent = list(range(n_seqs))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n_seqs):
                neighbors = I[lims[i]:lims[i + 1]]
                for j in neighbors:
                    if j != i:
                        union(i, int(j))

            # Assign global cluster IDs
            root_to_cluster = {}
            cluster_ids = np.zeros(n_seqs, dtype=int)
            for i in range(n_seqs):
                root = find(i)
                if root not in root_to_cluster:
                    root_to_cluster[root] = global_cluster_id
                    global_cluster_id += 1
                cluster_ids[i] = root_to_cluster[root]

            # Track which subjects appear in each cluster
            for i, seq in enumerate(valid_tcrs):
                cid = cluster_ids[i]
                if cid not in cluster_subject_sets:
                    cluster_subject_sets[cid] = set()
                cluster_subject_sets[cid].update(tcr_to_subjects[seq])

            # Store for prediction
            self._faiss_indices[length] = index
            self._cluster_assignments[length] = cluster_ids

            print(f"  Length {length}: {n_seqs} CDR3s -> "
                  f"{len(root_to_cluster)} clusters")

        print(f"Total clusters across all lengths: {global_cluster_id}")

        # Step 3: Fisher's exact test per cluster
        print("Computing p-values for all clusters...")
        cluster_pvalues = {}
        subject_labels = list(training_labels)

        for cid, subj_set in tqdm(cluster_subject_sets.items(),
                                   desc="Computing p-values"):
            pos_present = sum(1 for s in subj_set if subject_labels[s] == 1)
            neg_present = sum(1 for s in subj_set if subject_labels[s] == 0)

            # Require cluster to appear in at least 2 subjects
            if (pos_present + neg_present) < 2:
                continue

            a = pos_present
            b = neg_present
            c = total_pos - pos_present
            d = total_neg - neg_present

            table = [[a, b], [c, d]]
            _, p_value = fisher_exact(table, alternative='greater')
            cluster_pvalues[cid] = p_value

        self._cluster_stats_cache = {
            'cluster_pvalues': cluster_pvalues,
            'total_pos': total_pos,
            'total_neg': total_neg,
            'training_files': training_files,
            'training_labels': training_labels
        }

        print(f"Computed p-values for {len(cluster_pvalues)} clusters "
              f"(with incidence >= 2).")
        return self._cluster_stats_cache

    def select_diagnostic_clusters_from_cache(self, p_value_threshold=None):
        """
        Select diagnostic clusters from precomputed statistics.
        This is very fast as it just filters cached p-values.

        Args:
            p_value_threshold: P-value cutoff (uses self.p_value_threshold if None)

        Returns:
            Set of diagnostic cluster IDs
        """
        if self._cluster_stats_cache is None:
            raise ValueError("Cluster statistics not computed. "
                             "Call compute_cluster_statistics first.")

        if p_value_threshold is None:
            p_value_threshold = self.p_value_threshold

        pvalues = self._cluster_stats_cache['cluster_pvalues']
        self.diagnostic_clusters = {
            cid for cid, pval in pvalues.items()
            if pval < p_value_threshold
        }
        return self.diagnostic_clusters

    def _count_diagnostic_matches(self, file_path):
        """
        Count how many of a patient's CDR3s match diagnostic clusters.

        For each patient CDR3, finds the nearest training TCR via FAISS.
        If within the isometric distance threshold and the matched training TCR
        belongs to a diagnostic cluster, it counts as a match.

        Args:
            file_path: Path to patient repertoire file

        Returns:
            Tuple (n, k) where n = total unique CDR3s, k = CDR3s in diagnostic clusters
        """
        sequences = self.load_repertoire(file_path)
        n = len(sequences)
        k = 0

        # Group patient CDR3s by length for batched FAISS queries
        length_groups = {}
        for seq in sequences:
            length = len(seq)
            if length not in self._faiss_indices:
                continue
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(seq)

        for length, seqs in length_groups.items():
            encoded_matrix, valid_idx = self._encode_batch(seqs)
            if len(valid_idx) == 0:
                continue

            effective_thr = self._get_effective_threshold(length)

            # Batch FAISS query: find 1 nearest neighbor for each CDR3
            D, I = self._faiss_indices[length].search(encoded_matrix, 1)

            cluster_ids = self._cluster_assignments[length]

            for i in range(len(valid_idx)):
                dist = D[i, 0]
                if dist <= effective_thr:
                    matched_cluster = cluster_ids[I[i, 0]]
                    if matched_cluster in self.diagnostic_clusters:
                        k += 1

        return n, k

    def _neg_log_likelihood(self, params, n_values, k_values):
        """
        Negative Log Likelihood for the Beta-Binomial distribution.

        Args:
            params: Tuple of (alpha, beta) parameters
            n_values: Array of total sequence counts per subject
            k_values: Array of diagnostic match counts per subject

        Returns:
            Negative log likelihood value
        """
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf

        term1 = betaln(alpha, beta)
        term2 = betaln(k_values + alpha, n_values - k_values + beta)
        log_likelihood = np.sum(term2) - len(n_values) * term1

        return -log_likelihood

    def train_beta_binomial_model(self, training_files, training_labels):
        """
        Train the Beta-Binomial generative model.

        For each training subject, computes n (total unique CDR3s) and
        k (CDR3s matching diagnostic clusters), then fits alpha/beta
        parameters for Diseased and Healthy classes separately.

        Args:
            training_files: List of file paths to patient .tsv files
            training_labels: List of integers (1 for Diseased, 0 for Healthy)

        Returns:
            Dictionary with fitted model parameters
        """
        if not self.diagnostic_clusters:
            raise ValueError("No diagnostic clusters found. "
                             "Run select_diagnostic_clusters_from_cache first.")

        print("--- Training Beta-Binomial Model ---")

        data = {0: {'n': [], 'k': []}, 1: {'n': [], 'k': []}}

        for file_path, label in tqdm(zip(training_files, training_labels),
                                      total=len(training_files),
                                      desc="Computing features"):
            n, k_val = self._count_diagnostic_matches(file_path)
            data[label]['n'].append(n)
            data[label]['k'].append(k_val)

        initial_guess = [1.0, 1000.0]
        self.model_params = {}

        for label, label_name in [(0, 'Healthy'), (1, 'Diseased')]:
            n_vals = np.array(data[label]['n'])
            k_vals = np.array(data[label]['k'])

            result = minimize(
                self._neg_log_likelihood,
                initial_guess,
                args=(n_vals, k_vals),
                method='L-BFGS-B',
                bounds=((1e-5, None), (1e-5, None))
            )

            self.model_params[label] = {
                'alpha': result.x[0],
                'beta': result.x[1],
                'count': len(n_vals)
            }
            print(f"Fit {label_name}: alpha={result.x[0]:.4f}, beta={result.x[1]:.4f}")

        return self.model_params

    def predict_diagnosis(self, file_path):
        """
        Predict disease status for a new patient.

        Encodes patient CDR3s, queries the FAISS index to find matches
        to diagnostic clusters, and computes the Beta-Binomial posterior
        probability of disease.

        Args:
            file_path: Path to the patient's repertoire TSV file

        Returns:
            Dictionary with prediction results including:
            - n: Total unique CDR3s in patient
            - k: Number of CDR3s matching diagnostic clusters
            - diagnostic_clusters_count: Total diagnostic clusters in model
            - probability_positive: Posterior probability of disease
            - diagnosis: 'Diseased' or 'Healthy'
        """
        if not self.model_params:
            raise ValueError("Model not trained.")

        n, k = self._count_diagnostic_matches(file_path)

        # Beta-Binomial posterior (same formulation as Emerson 2017)
        log_probs = {}
        total_subjects = self.model_params[0]['count'] + self.model_params[1]['count']

        for label in [0, 1]:
            alpha = self.model_params[label]['alpha']
            beta_param = self.model_params[label]['beta']
            count = self.model_params[label]['count']

            log_likelihood = betaln(k + alpha, n - k + beta_param) - \
                             betaln(alpha, beta_param)
            log_prior = np.log(count + 1) - np.log(total_subjects + 2)
            log_probs[label] = log_likelihood + log_prior

        log_diff = log_probs[0] - log_probs[1]
        prob_positive = 1 / (1 + np.exp(log_diff))

        return {
            'n': n,
            'k': k,
            'diagnostic_clusters_count': len(self.diagnostic_clusters),
            'probability_positive': prob_positive,
            'diagnosis': 'Diseased' if prob_positive > 0.5 else 'Healthy'
        }


# --- Usage Example ---

if __name__ == "__main__":
    print("GIANA 2020 TCR Disease Classification Model")
    print("=" * 60)
    print("\nThis module contains the core model implementation.")
    print("For evaluation with cross-validation, use:")
    print("  from evals.giana_2020_disease_classification import GIANA2020Evaluator")
    print("\nBasic usage example:")
    print("  model = GIANA_Classifier(iso_threshold=7, p_value_threshold=1e-4)")
    print("  model.compute_cluster_statistics(train_files, train_labels)")
    print("  model.select_diagnostic_clusters_from_cache()")
    print("  model.train_beta_binomial_model(train_files, train_labels)")
    print("  result = model.predict_diagnosis('patient.tsv.gz')")
