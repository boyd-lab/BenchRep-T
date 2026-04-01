"""
Emerson 2017 CMV Immunosequencing Model.

Implementation of the CMV (Cytomegalovirus) classification model from:
Emerson et al., "Immunosequencing identifies signatures of cytomegalovirus 
exposure history and HLA-mediated effects on the T cell repertoire" (2017)

This module contains the core model for:
- Identifying CMV-associated TCR sequences using Fisher's exact test
- Training a Beta-Binomial generative model
- Predicting CMV status for new patients
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from scipy.special import betaln
from scipy.optimize import minimize
from tqdm import tqdm
from utils.repertoire_io import load_raw_repertoire


class CMV_Immunosequencing_Model:
    """
    CMV Immunosequencing Model based on Emerson et al. 2017.
    
    This model identifies diagnostic TCR sequences associated with CMV status
    and uses a Beta-Binomial generative model for classification.
    """
    
    def __init__(self, p_value_threshold=1e-4, sequence_col='cdr3_aa',
                 v_col='v_call', j_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None,
                 v_gene_harmonizer=None, j_gene_harmonizer=None,
                 ignore_allele=False):
        """
        Initialize the model with a p-value threshold for feature selection.
        Paper uses 1e-4 as optimized via cross-validation[cite: 52].

        Args:
            p_value_threshold: P-value cutoff for Fisher's exact test (default: 1e-4)
            sequence_col: Column name containing CDR3 amino acid sequences (default: 'cdr3_aa')
            v_col: Column name containing V gene calls (default: 'v_call')
            j_col: Column name containing J gene calls (default: 'j_call')
            subsample_fraction: Fraction of reads to keep for depth simulation (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling (default: 42)
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set)
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
                         When set, overrides subsample_n/fraction/seed.
            v_gene_harmonizer: Optional callable applied to V gene values after loading
                               (e.g. adaptive_to_airr for cross-dataset evaluation).
            j_gene_harmonizer: Optional callable applied to J gene values after loading.
            ignore_allele: If True, strip allele designations (*XX) from V/J gene
                           names when building TCR tuples, enabling gene-level matching
                           across datasets with different allele conventions.
        """
        self.p_value_threshold = p_value_threshold
        self.sequence_col = sequence_col
        self.v_col = v_col
        self.j_col = j_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.v_gene_harmonizer = v_gene_harmonizer
        self.j_gene_harmonizer = j_gene_harmonizer
        self.ignore_allele = ignore_allele
        self.diagnostic_tcrs = set()
        self.model_params = {}  # Will store alphas, betas, and priors
        self._repertoire_cache = {}  # Cache for loaded repertoire data
        self._tcr_stats_cache = None  # Cache for TCR p-values
        
    def load_repertoire(self, file_path, use_cache=True):
        """
        Loads a single patient's repertoire from a TSV file.
        
        Args:
            file_path: Path to the repertoire TSV file (supports .tsv and .tsv.gz)
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            Set of unique TCR sequences
        """
        # Check cache first
        if use_cache and file_path in self._repertoire_cache:
            return self._repertoire_cache[file_path]

        indices = None
        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
            indices = self.indices_map.get(rep_id)
        df = load_raw_repertoire(file_path, self.subsample_n, self.subsample_fraction,
                                 self.subsample_seed, subsample_indices=indices)
        if df.empty:
            return set()

        for col in [self.sequence_col, self.v_col, self.j_col]:
            if col not in df.columns:
                print(f"Error loading {file_path}: Column '{col}' not found")
                return set()

        # Return unique (v_call, cdr3_aa, j_call) tuples — matching the paper's definition
        # of a unique TCRβ as a combination of V gene, CDR3 amino acid sequence, and J gene.
        df = df[[self.v_col, self.sequence_col, self.j_col]].dropna()

        # Apply gene name harmonization if configured (e.g. Adaptive → AIRR)
        if self.v_gene_harmonizer:
            df[self.v_col] = df[self.v_col].map(self.v_gene_harmonizer)
        if self.j_gene_harmonizer:
            df[self.j_col] = df[self.j_col].map(self.j_gene_harmonizer)

        # Strip allele designations for gene-level matching across datasets
        if self.ignore_allele:
            from utils.gene_harmonization import strip_allele
            df[self.v_col] = df[self.v_col].map(strip_allele)
            df[self.j_col] = df[self.j_col].map(strip_allele)

        sequences = set(zip(df[self.v_col], df[self.sequence_col], df[self.j_col]))

        if use_cache:
            self._repertoire_cache[file_path] = sequences
        return sequences
    
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
        """Clear the repertoire cache to free memory."""
        self._repertoire_cache = {}
        self._tcr_stats_cache = None
    
    def compute_tcr_statistics(self, training_files, training_labels):
        """
        Precompute TCR counts and p-values for all TCRs in training data.
        This is expensive but only needs to be done once per fold.
        
        Args:
            training_files: List of file paths to patient .tsv files.
            training_labels: List of integers (1 for Diseased, 0 for Healthy).
            
        Returns:
            Dictionary with TCR statistics including p-values
        """
        print("--- Computing TCR Statistics (one-time) ---")
        tcr_counts = {}  # {tcr_sequence: {'pos': count, 'neg': count}}
        
        total_pos_subjects = sum(training_labels)
        total_neg_subjects = len(training_labels) - total_pos_subjects
        
        print(f"Analyzing {len(training_files)} subjects ({total_pos_subjects} Pos, {total_neg_subjects} Neg)...")

        for file_path, label in tqdm(zip(training_files, training_labels), 
                                      total=len(training_files), 
                                      desc="Building TCR counts"):
            unique_seqs = self.load_repertoire(file_path)
            
            for seq in unique_seqs:
                if seq not in tcr_counts:
                    tcr_counts[seq] = {'pos': 0, 'neg': 0}
                
                if label == 1:
                    tcr_counts[seq]['pos'] += 1
                else:
                    tcr_counts[seq]['neg'] += 1

        print(f"Total unique TCRs found: {len(tcr_counts)}")
        print("Computing p-values for all TCRs...")
        
        # Compute p-values for all TCRs (this is the expensive part)
        tcr_pvalues = {}
        for tcr, counts in tqdm(tcr_counts.items(), desc="Computing p-values"):
            pos_present = counts['pos']
            neg_present = counts['neg']
            
            # Paper Filter: "subject incidence of at least two" 
            if (pos_present + neg_present) < 2:
                continue
            
            a = pos_present
            b = neg_present
            c = total_pos_subjects - pos_present
            d = total_neg_subjects - neg_present
            
            table = [[a, b], [c, d]]
            _, p_value = fisher_exact(table, alternative='greater')
            
            tcr_pvalues[tcr] = p_value
        
        self._tcr_stats_cache = {
            'tcr_pvalues': tcr_pvalues,
            'total_pos_subjects': total_pos_subjects,
            'total_neg_subjects': total_neg_subjects,
            'training_files': training_files,
            'training_labels': training_labels
        }
        
        print(f"Computed p-values for {len(tcr_pvalues)} TCRs (with incidence >= 2).")
        return self._tcr_stats_cache
    
    def select_diagnostic_tcrs_from_cache(self, p_value_threshold=None):
        """
        Select diagnostic TCRs from precomputed statistics using a p-value threshold.
        This is very fast as it just filters the cached p-values.
        
        Args:
            p_value_threshold: P-value cutoff (uses self.p_value_threshold if None)
            
        Returns:
            Set of diagnostic TCR sequences
        """
        if self._tcr_stats_cache is None:
            raise ValueError("TCR statistics not computed. Call compute_tcr_statistics first.")
        
        if p_value_threshold is None:
            p_value_threshold = self.p_value_threshold
        
        tcr_pvalues = self._tcr_stats_cache['tcr_pvalues']
        
        diagnostic_candidates = [
            tcr for tcr, pval in tcr_pvalues.items() 
            if pval < p_value_threshold
        ]
        
        self.diagnostic_tcrs = set(diagnostic_candidates)
        return self.diagnostic_tcrs

    def identify_diagnostic_tcrs(self, training_files, training_labels):
        """
        Step 1: Fisher's Exact Test to identify CMV-associated TCRs.
        
        Args:
            training_files: List of file paths to patient .tsv files.
            training_labels: List of integers (1 for Diseased, 0 for Healthy) corresponding to files.
            
        Returns:
            Set of diagnostic TCR sequences
        """
        print("--- Step 1: Loading Training Data ---")
        # 1. Load all data into memory
        # We need to map every unique TCR to which subjects have it
        tcr_counts = {}  # {tcr_sequence: {'pos': count, 'neg': count}}
        
        total_pos_subjects = sum(training_labels)
        total_neg_subjects = len(training_labels) - total_pos_subjects
        
        print(f"Analyzing {len(training_files)} subjects ({total_pos_subjects} Pos, {total_neg_subjects} Neg)...")

        for file_path, label in tqdm(zip(training_files, training_labels), 
                                      total=len(training_files), 
                                      desc="Loading repertoires"):
            unique_seqs = self.load_repertoire(file_path)
            
            for seq in unique_seqs:
                if seq not in tcr_counts:
                    tcr_counts[seq] = {'pos': 0, 'neg': 0}
                
                if label == 1:
                    tcr_counts[seq]['pos'] += 1
                else:
                    tcr_counts[seq]['neg'] += 1

        print(f"Total unique TCRs found: {len(tcr_counts)}")
        print("--- Step 2: Running Fisher's Exact Test ---")
        
        diagnostic_candidates = []
        
        # 2. Iterate through all TCRs and apply filters
        for tcr, counts in tcr_counts.items():
            pos_present = counts['pos']
            neg_present = counts['neg']
            
            # Paper Filter: "subject incidence of at least two" 
            if (pos_present + neg_present) < 2:
                continue
                
            # Contingency Table Construction [cite: 461, 560]
            #           | Dis+ | Dis- |
            # -------------------------
            # Present   |  a   |  b   |
            # Absent    |  c   |  d   |
            
            a = pos_present
            b = neg_present
            c = total_pos_subjects - pos_present
            d = total_neg_subjects - neg_present
            
            table = [[a, b], [c, d]]
            
            # One-tailed Fisher's exact test (alternative='greater') 
            # to find enrichment in diseased subjects [cite: 464]
            _, p_value = fisher_exact(table, alternative='greater')
            
            if p_value < self.p_value_threshold:
                diagnostic_candidates.append(tcr)

        self.diagnostic_tcrs = set(diagnostic_candidates)
        print(f"Identified {len(self.diagnostic_tcrs)} diagnostic TCRs (P < {self.p_value_threshold}).")
        return self.diagnostic_tcrs

    def _neg_log_likelihood(self, params, n_values, k_values):
        """
        Negative Log Likelihood for the Beta-Binomial distribution.
        Used for optimization.
        
        Equation derived from[cite: 495]:
        L(alpha, beta) = Sum( log B(k + alpha, n - k + beta) ) - N * log B(alpha, beta)
        
        Args:
            params: Tuple of (alpha, beta) parameters
            n_values: Array of total sequence counts per subject
            k_values: Array of diagnostic sequence counts per subject
            
        Returns:
            Negative log likelihood value
        """
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf  # Constraints
        
        # term1 = log B(alpha, beta)
        term1 = betaln(alpha, beta)
        
        # term2 = log B(k + alpha, n - k + beta)
        term2 = betaln(k_values + alpha, n_values - k_values + beta)
        
        # L = Sum(term2) - N * term1
        log_likelihood = np.sum(term2) - len(n_values) * term1
        
        # Return negative for minimization
        return -log_likelihood

    def train_beta_binomial_model(self, training_files, training_labels):
        """
        Step 2: Train the generative Beta-Binomial model.
        
        Calculates n (total sequences) and k (diagnostic sequences) for each subject,
        then fits alpha/beta parameters for Diseased and Healthy populations separately.
        
        Args:
            training_files: List of file paths to patient .tsv files.
            training_labels: List of integers (1 for Diseased, 0 for Healthy) corresponding to files.
            
        Returns:
            Dictionary with fitted model parameters
        """
        if not self.diagnostic_tcrs:
            raise ValueError("No diagnostic TCRs found. Run identify_diagnostic_tcrs first.")

        print("--- Step 3: Training Beta-Binomial Model ---")
        
        # Arrays to store (n, k) for each class
        data = {0: {'n': [], 'k': []}, 1: {'n': [], 'k': []}}
        
        for file_path, label in tqdm(zip(training_files, training_labels),
                                      total=len(training_files),
                                      desc="Computing features"):
            unique_seqs = self.load_repertoire(file_path)
            
            n_i = len(unique_seqs)  # Total unique TCRBs [cite: 477]
            # Intersection size: count how many diagnostic TCRs are in this subject
            k_i = len(unique_seqs.intersection(self.diagnostic_tcrs))  # [cite: 477]
            
            data[label]['n'].append(n_i)
            data[label]['k'].append(k_i)

        # Optimize parameters for both classes
        # Initial guesses for alpha, beta
        initial_guess = [1.0, 1000.0] 
        self.model_params = {}
        
        for label, label_name in [(0, 'Healthy'), (1, 'Diseased')]:
            n_vals = np.array(data[label]['n'])
            k_vals = np.array(data[label]['k'])
            
            # Optimize using L-BFGS-B (gradient descent) 
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
                'count': len(n_vals)  # Store N for prior calculation
            }
            print(f"Fit {label_name}: alpha={result.x[0]:.4f}, beta={result.x[1]:.4f}")
            
        return self.model_params

    def predict_diagnosis(self, file_path):
        """
        Step 3: Diagnose a new patient.
        
        Calculates the posterior probability P(Diseased | n, k)
        
        Args:
            file_path: Path to the patient's repertoire TSV file
            
        Returns:
            Dictionary with prediction results including:
            - n: Total unique TCRs in patient
            - k: Number of diagnostic TCRs found
            - diagnostic_tcrs_count: Total diagnostic TCRs in model
            - probability_positive: Posterior probability of disease
            - diagnosis: 'Diseased' or 'Healthy'
        """
        if not self.model_params:
            raise ValueError("Model not trained.")
            
        # 1. Calculate phenotype burden for the new subject
        unique_seqs = self.load_repertoire(file_path)
        n_prime = len(unique_seqs)
        k_prime = len(unique_seqs.intersection(self.diagnostic_tcrs))
        
        # 2. Calculate Log Posterior Odds [cite: 509-511]
        # P(c|data) propto P(data|c) * P(c)
        
        log_probs = {}
        
        total_subjects = self.model_params[0]['count'] + self.model_params[1]['count']
        
        for label in [0, 1]:
            alpha = self.model_params[label]['alpha']
            beta_param = self.model_params[label]['beta']
            count = self.model_params[label]['count']
            
            # Log Likelihood: log Beta-Binomial(k | n, alpha, beta)
            # log(nCk) is constant for both classes, so we can ignore it for comparison
            # We only need log B(k+alpha, n-k+beta) - log B(alpha, beta)
            log_likelihood = betaln(k_prime + alpha, n_prime - k_prime + beta_param) - \
                             betaln(alpha, beta_param)
            
            # Log Prior: Laplace smoothing (N_c + 1) / (N + 2) 
            log_prior = np.log(count + 1) - np.log(total_subjects + 2)
            
            log_probs[label] = log_likelihood + log_prior

        # 3. Convert Log Odds to Probability
        # P(Diseased) = 1 / (1 + exp(log_prob_neg - log_prob_pos))
        log_diff = log_probs[0] - log_probs[1]
        prob_cmv_pos = 1 / (1 + np.exp(log_diff))
        
        return {
            'n': n_prime,
            'k': k_prime,
            'diagnostic_tcrs_count': len(self.diagnostic_tcrs),
            'probability_positive': prob_cmv_pos,
            'diagnosis': 'Diseased' if prob_cmv_pos > 0.5 else 'Healthy'
        }


# --- Usage Example ---

if __name__ == "__main__":
    print("Emerson 2017 CMV Immunosequencing Model")
    print("=" * 60)
    print("\nThis module contains the core model implementation.")
    print("For evaluation with cross-validation, use:")
    print("  from evals.emerson_2017_disease_classification import Emerson2017Evaluator")
    print("\nBasic usage example:")
    print("  model = CMV_Immunosequencing_Model(p_value_threshold=1e-4, sequence_col='cdr3_aa')")
    print("  model.identify_diagnostic_tcrs(train_files, train_labels)")
    print("  model.train_beta_binomial_model(train_files, train_labels)")
    print("  result = model.predict_diagnosis('patient.tsv.gz')")
    
    # Example (commented out as files don't exist):
    # train_files = ['patient1.tsv.gz', 'patient2.tsv.gz', ...]
    # train_labels = [1, 0, ...] 
    
    # model = CMV_Immunosequencing_Model(p_value_threshold=1e-4, sequence_col='cdr3_aa')
    # model.identify_diagnostic_tcrs(train_files, train_labels)
    # model.train_beta_binomial_model(train_files, train_labels)
    # result = model.predict_diagnosis('new_patient.tsv.gz')
    # print(result)
