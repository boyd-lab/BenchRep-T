"""
Ostmeyer 2019 MIL-TCR Classification Model.

Implementation of the Multiple Instance Learning (MIL) model from:
Ostmeyer et al., "Biophysicochemical motifs in T-cell receptor sequences
distinguish repertoires from tumor-infiltrating lymphocyte and adjacent
healthy tissue" (2019)

This module contains the core model for:
- Extracting 4-mer motifs from CDR3 sequences
- Encoding amino acids using Atchley factors
- Training a logistic regression model with MIL aggregation
- Predicting disease status for new patients
"""

import os
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from utils.repertoire_io import load_raw_repertoire


# ATCHLEY FACTORS (Reference: Atchley et al., 2005)
# Each amino acid maps to 5 numerical factors representing:
# 1. Polarity, 2. Secondary Structure, 3. Molecular Volume, 4. Codon Diversity, 5. Electrostatic Charge
# Values normalized to mean=0, std=1 as per the paper
ATCHLEY_FACTORS = {
    'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
    'D': [1.050, 0.302, -3.656, -0.259, -3.242],
    'E': [1.357, -1.453, 1.477, 0.113, -0.837],
    'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
    'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.440, 2.897],
    'S': [-0.228, 1.399, -4.760, 0.670, -0.033],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [0.260, 0.830, 3.097, -0.838, 1.512]
}


class MIL_TCR_Classifier:
    """
    MIL-TCR Classifier based on Ostmeyer et al. 2019.

    This model uses Multiple Instance Learning with logistic regression
    on 4-mer motifs encoded using Atchley factors to classify disease status.

    Clone sizes are derived by counting rows with the same cdr3_aa sequence,
    following the paper's footnote: "We treat TCRb sequences with identical
    CDR3 sequences as being the same TCRb sequence."

    Abundance is computed per the paper's Equation A or B (configurable):
      - 'A': 4-mer relative abundance (sum of clone sizes for all CDR3s
             containing the 4-mer, divided by total count of all 4-mers)
      - 'B': TCRb relative abundance (max clone relative abundance among
             all CDR3s containing the 4-mer)

    No L1/L2 regularization is applied, as the paper explicitly states it
    worsened performance.

    Multiple random restarts are used; the run with the lowest training loss
    is kept (default: 250,000 restarts, matching the paper's best models).
    """

    def __init__(self, n_restarts=200, lbfgsb_maxiter=1000,
                 abundance_method='A',
                 sequence_col='cdr3_aa', min_cdr3_length=10,
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None):
        """
        Initialize the model.

        Args:
            n_restarts: Number of random restarts for optimization (default: 2000).
                        L-BFGS-B converges in ~50-200 gradient evaluations per
                        restart, so 2000 restarts gives broad landscape coverage
                        at a fraction of the paper's compute cost.
            lbfgsb_maxiter: Maximum L-BFGS-B iterations per restart (default: 1000).
            abundance_method: 'A' for 4-mer relative abundance (Equation A in
                              paper) or 'B' for TCRb relative abundance
                              (Equation B). Default: 'A'.
            sequence_col: Column name containing CDR3 amino acid sequences
                          (default: 'cdr3_aa')
            min_cdr3_length: Minimum CDR3 length to include (default: 10,
                             which is the minimum to yield at least one 4-mer
                             after trimming 3 residues from each end)
            subsample_fraction: Fraction of rows to keep for depth simulation
                                (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling
                            (default: 7)
            subsample_n: Absolute number of reads to keep (overrides
                         subsample_fraction if set)
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
                         When set, overrides subsample_n/fraction/seed.
        """
        self.n_restarts = n_restarts
        self.lbfgsb_maxiter = lbfgsb_maxiter
        self.abundance_method = abundance_method
        self.sequence_col = sequence_col
        self.min_cdr3_length = min_cdr3_length
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map

        self.atchley = ATCHLEY_FACTORS

        # Model parameters: 4 residues * 5 factors + 1 abundance weight + 1 bias
        # W1-W20: Atchley weights, W21: log-abundance weight, b0: bias
        self.weights = None  # Will be (21,) vector
        self.bias = None     # Will be scalar

        self._repertoire_cache = {}  # Cache for loaded repertoire data
        self._features_cache = {}    # Cache for extracted 4-mer features

    def load_repertoire(self, file_path, use_cache=True):
        """
        Load a single patient's repertoire from a TSV file.

        Args:
            file_path: Path to the repertoire TSV file (supports .tsv and .tsv.gz)
            use_cache: Whether to use cached data if available (default: True)

        Returns:
            DataFrame with repertoire data
        """
        if use_cache and file_path in self._repertoire_cache:
            return self._repertoire_cache[file_path]

        indices = None
        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
            indices = self.indices_map.get(rep_id)
        df = load_raw_repertoire(file_path, self.subsample_n, self.subsample_fraction,
                                 self.subsample_seed, subsample_indices=indices)
        if df.empty:
            return df

        if self.sequence_col not in df.columns:
            print(f"Error loading {file_path}: Column '{self.sequence_col}' not found")
            return pd.DataFrame()

        if use_cache:
            self._repertoire_cache[file_path] = df
        return df

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
        self._features_cache = {}

    def save(self, path):
        """Save fitted weights and configuration without repertoire caches."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = self.__dict__.copy()
        state['_repertoire_cache'] = {}
        state['_features_cache'] = {}
        with open(path, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load a fitted model saved by :meth:`save` for inference."""
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        obj = cls()
        obj.__dict__.update(state)
        return obj

    def _build_fourmer_data(self, file_path):
        """
        Build 4-mer features, log-abundances, and 4-mer string list for a repertoire.

        Clone sizes are computed by counting rows with identical cdr3_aa sequences,
        per the paper's footnote. Features are one entry per unique 4-mer per sample.

        Abundance follows the paper's Equation A or B (self.abundance_method):
          A: RA = C_4mer / T_4mer, where C_4mer = sum of clone sizes for all CDR3s
             containing the 4-mer, and T_4mer = sum of C_4mer over all unique 4-mers.
          B: RA = max(C_TCRb / T_TCRb) over all CDR3s containing the 4-mer, where
             T_TCRb = total clone count across all valid CDR3s.

        Returns:
            Tuple of (feature_matrix, log_abundances, fourmer_strings) or
            (empty arrays, empty array, []) if no valid data.
        """
        df = self.load_repertoire(file_path)
        if df.empty:
            return np.array([]).reshape(0, 20), np.array([]), []

        # Clone sizes: number of rows per unique cdr3_aa
        cdr3_counts = df[self.sequence_col].value_counts()

        # Filter by minimum CDR3 length
        valid_cdr3s = {
            cdr3: count for cdr3, count in cdr3_counts.items()
            if isinstance(cdr3, str) and len(cdr3) >= self.min_cdr3_length
        }

        if not valid_cdr3s:
            return np.array([]).reshape(0, 20), np.array([]), []

        total_templates = sum(valid_cdr3s.values())

        # Build mapping: unique 4-mer -> list of clone sizes for CDR3s containing it
        # Insertion order preserved (Python 3.7+), so iteration order is consistent.
        fourmer_to_counts = {}
        for cdr3, count in valid_cdr3s.items():
            core = cdr3[3:-3]  # exclude first 3 and last 3 residues
            for i in range(len(core) - 3):
                fourmer = core[i:i + 4]
                if any(aa not in self.atchley for aa in fourmer):
                    continue
                if fourmer not in fourmer_to_counts:
                    fourmer_to_counts[fourmer] = []
                fourmer_to_counts[fourmer].append(count)

        if not fourmer_to_counts:
            return np.array([]).reshape(0, 20), np.array([]), []

        all_features = []
        all_log_abundances = []
        fourmer_strings = []

        if self.abundance_method == 'A':
            # Equation A: 4-mer relative abundance
            # C_4mer = sum of clone sizes for all CDR3s containing the 4-mer
            # T_4mer = sum of C_4mer over all unique 4-mers
            # RA = C_4mer / T_4mer
            c4mer = {fm: sum(counts) for fm, counts in fourmer_to_counts.items()}
            t4mer = sum(c4mer.values())

            for fm, c in c4mer.items():
                features = []
                for aa in fm:
                    features.extend(self.atchley[aa])
                ra = c / t4mer
                log_ra = np.log(ra) if ra > 0 else -10.0
                all_features.append(features)
                all_log_abundances.append(log_ra)
                fourmer_strings.append(fm)

        else:
            # Equation B: TCRb relative abundance
            # T_TCRb = total clone count across all valid CDR3s
            # RA = max(C_TCRb / T_TCRb) over all CDR3s containing the 4-mer
            for fm, counts in fourmer_to_counts.items():
                features = []
                for aa in fm:
                    features.extend(self.atchley[aa])
                max_count = max(counts)
                ra = max_count / total_templates if total_templates > 0 else 0.0
                log_ra = np.log(ra) if ra > 0 else -10.0
                all_features.append(features)
                all_log_abundances.append(log_ra)
                fourmer_strings.append(fm)

        return np.array(all_features), np.array(all_log_abundances), fourmer_strings

    def extract_4mer_features(self, file_path, use_cache=True):
        """
        Extract 4-mer feature matrix and log-abundance vector for a repertoire.

        Returns:
            Tuple of (feature_matrix, log_abundances) where:
            - feature_matrix: (N, 20) array of Atchley-encoded 4-mers
            - log_abundances: (N,) array of log relative abundances
        """
        if use_cache and file_path in self._features_cache:
            features, log_abundances, _ = self._features_cache[file_path]
            return features, log_abundances

        features, log_abundances, fourmer_strings = self._build_fourmer_data(file_path)

        if use_cache:
            self._features_cache[file_path] = (features, log_abundances, fourmer_strings)

        return features, log_abundances

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def _compute_instance_scores(self, features, abundances):
        """
        Compute logistic regression scores for all 4-mer instances.

        Args:
            features: (N, 20) array of Atchley features
            abundances: (N,) array of log abundances

        Returns:
            (N,) array of instance probabilities
        """
        if len(features) == 0:
            return np.array([0.5])
        combined_features = np.column_stack([features, abundances])
        logits = combined_features @ self.weights + self.bias
        return self._sigmoid(logits)

    def _compute_bag_probability(self, instance_scores):
        """
        MIL aggregation: bag probability = max instance score.

        Args:
            instance_scores: (N,) array of instance probabilities

        Returns:
            Bag probability (scalar)
        """
        if len(instance_scores) == 0:
            return 0.5
        return np.max(instance_scores)

    def _compute_loss_and_gradient(self, params, training_data, labels):
        """
        Compute cross-entropy loss and gradient for the MIL model.

        No regularization is applied (paper explicitly avoids L1/L2).
        Gradient flows only through the max-scoring 4-mer (the "witness").

        Args:
            params: Flattened parameters vector (weights[0:21] + bias[21])
            training_data: List of (features, log_abundances) tuples
            labels: List of binary labels (1=disease, 0=healthy)

        Returns:
            Tuple of (loss, gradient)
        """
        weights = params[:21]
        bias = params[21]

        total_loss = 0.0
        total_grad = np.zeros(22)

        for (features, abundances), label in zip(training_data, labels):
            if len(features) == 0:
                continue

            combined_features = np.column_stack([features, abundances])
            logits = combined_features @ weights + bias
            instance_probs = self._sigmoid(logits)

            # MIL: bag probability = max instance score
            max_idx = np.argmax(instance_probs)
            bag_prob = instance_probs[max_idx]

            eps = 1e-10
            bag_prob_clipped = np.clip(bag_prob, eps, 1 - eps)
            loss = -(label * np.log(bag_prob_clipped) +
                     (1 - label) * np.log(1 - bag_prob_clipped))
            total_loss += loss

            # Gradient through max: only the witness (max-scoring 4-mer) contributes.
            # d(loss)/d(logit) = sigmoid(logit) - label = bag_prob - label
            d_loss_d_logit = bag_prob - label
            witness_features = combined_features[max_idx]
            total_grad[:21] += d_loss_d_logit * witness_features
            total_grad[21] += d_loss_d_logit

        n_samples = len(labels)
        total_loss /= n_samples
        total_grad /= n_samples

        # No L1/L2 regularization (paper: "both worsened model performance")

        return total_loss, total_grad

    def train(self, training_files, training_labels):
        """
        Train the MIL-TCR model using multiple random restarts.

        Weight initialization follows the paper:
          - W1-W20 (Atchley weights) ~ N(0, 1/n_features)
          - W21 (log-abundance weight) = 0
          - b0 (bias) ~ N(0, 1/n_features)
        The restart with the lowest training loss is kept.

        Args:
            training_files: List of file paths to patient .tsv files
            training_labels: List of integers (1 for Diseased, 0 for Healthy)

        Returns:
            Dictionary with training results
        """
        print("--- Training MIL-TCR Classifier ---")

        print("Extracting 4-mer features from training data...")
        training_data = []
        valid_labels = []

        for file_path, label in tqdm(zip(training_files, training_labels),
                                     total=len(training_files),
                                     desc="Extracting features"):
            features, abundances = self.extract_4mer_features(file_path)
            if len(features) > 0:
                training_data.append((features, abundances))
                valid_labels.append(label)

        if len(training_data) == 0:
            raise ValueError("No valid training samples found!")

        n_features = 20  # 20 Atchley factors (paper initializes W ~ N(0, 1/n_features))

        print(f"Training on {len(training_data)} samples with {self.n_restarts} L-BFGS-B restarts...")

        best_loss = np.inf
        best_params = None

        for _ in tqdm(range(self.n_restarts), desc="Optimization restarts"):
            # Initialize per paper: W1-W20 and b0 ~ N(0, 1/n_features), W21=0
            params = np.zeros(22)
            params[:20] = np.random.randn(20) * (1.0 / n_features)
            params[20] = 0.0  # W21 (log-abundance weight) = 0
            params[21] = np.random.randn() * (1.0 / n_features)  # b0

            result = minimize(
                fun=lambda p: self._compute_loss_and_gradient(p, training_data, valid_labels),
                x0=params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': self.lbfgsb_maxiter},
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x.copy()

        self.weights = best_params[:21]
        self.bias = best_params[21]

        print(f"Training completed. Best loss across restarts: {best_loss:.4f}")

        train_preds = []
        for features, abundances in training_data:
            instance_scores = self._compute_instance_scores(features, abundances)
            bag_prob = self._compute_bag_probability(instance_scores)
            train_preds.append(1 if bag_prob >= 0.5 else 0)

        train_accuracy = np.mean(np.array(train_preds) == np.array(valid_labels))
        print(f"Training accuracy: {train_accuracy:.4f}")

        return {
            'best_loss': best_loss,
            'train_accuracy': train_accuracy,
            'n_samples': len(training_data),
            'n_restarts': self.n_restarts
        }

    def predict_diagnosis(self, file_path):
        """
        Predict disease status for a new patient.

        Args:
            file_path: Path to the patient's repertoire TSV file

        Returns:
            Dictionary with prediction results including:
            - n_instances: Number of unique 4-mer instances extracted
            - probability_positive: Probability of disease (max 4-mer score)
            - diagnosis: 'Diseased' or 'Healthy'
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call train() first.")

        features, abundances = self.extract_4mer_features(file_path)

        if len(features) == 0:
            return {
                'n_instances': 0,
                'probability_positive': 0.5,
                'diagnosis': 'Unknown'
            }

        instance_scores = self._compute_instance_scores(features, abundances)
        bag_prob = self._compute_bag_probability(instance_scores)

        return {
            'n_instances': len(features),
            'probability_positive': float(bag_prob),
            'diagnosis': 'Diseased' if bag_prob >= 0.5 else 'Healthy'
        }

    def get_diagnostic_motifs(self, file_path, threshold=0.5, top_k=10):
        """
        Get the highest-scoring 4-mer motifs from a repertoire.

        Args:
            file_path: Path to the patient's repertoire TSV file
            threshold: Score threshold for "diagnostic" motifs (default: 0.5)
            top_k: Number of top motifs to return (default: 10)

        Returns:
            DataFrame with top diagnostic motifs and their scores
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call train() first.")

        # Use cached data if available (includes fourmer_strings)
        if file_path in self._features_cache:
            features, log_abundances, fourmer_strings = self._features_cache[file_path]
        else:
            features, log_abundances, fourmer_strings = self._build_fourmer_data(file_path)
            self._features_cache[file_path] = (features, log_abundances, fourmer_strings)

        if len(features) == 0:
            return pd.DataFrame()

        instance_scores = self._compute_instance_scores(features, log_abundances)

        motifs = []
        for fm, score, log_ra in zip(fourmer_strings, instance_scores, log_abundances):
            motifs.append({
                '4mer': fm,
                'score': float(score),
                'log_abundance': float(log_ra)
            })

        if not motifs:
            return pd.DataFrame()

        motifs_df = pd.DataFrame(motifs).sort_values('score', ascending=False)
        return motifs_df[motifs_df['score'] >= threshold].head(top_k)


# --- Usage Example ---

if __name__ == "__main__":
    print("Ostmeyer 2019 MIL-TCR Classification Model")
    print("=" * 60)
    print("\nThis module contains the core model implementation.")
    print("For evaluation with cross-validation, use:")
    print("  from evals.ostmeyer_2019_disease_classification import Ostmeyer2019Evaluator")
    print("\nBasic usage example:")
    print("  model = MIL_TCR_Classifier(n_restarts=200, abundance_method='A')")
    print("  model.train(train_files, train_labels)")
    print("  result = model.predict_diagnosis('patient.tsv.gz')")
