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

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


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
    """
    
    def __init__(self, learning_rate=0.01, max_iter=100, reg_strength=0.01,
                 sequence_col='cdr3_aa', min_cdr3_length=10,
                 subsample_fraction=1.0, subsample_seed=7):
        """
        Initialize the model.

        Args:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            max_iter: Maximum iterations for optimization (default: 100)
            reg_strength: L2 regularization strength (default: 0.01)
            sequence_col: Column name containing TCR sequences (default: 'cdr3_aa')
            min_cdr3_length: Minimum CDR3 length to include (default: 10)
            subsample_fraction: Fraction of reads to keep for depth simulation (default: 1.0)
            subsample_seed: Random seed for reproducible subsampling (default: 42)
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_strength = reg_strength
        self.sequence_col = sequence_col
        self.min_cdr3_length = min_cdr3_length
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        
        self.atchley = ATCHLEY_FACTORS
        
        # Model parameters: 4 residues * 5 factors + 1 abundance weight + 1 bias
        # Total: 22 parameters
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
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            if self.sequence_col not in df.columns:
                raise ValueError(f"Column '{self.sequence_col}' not found in {file_path}")
            # Subsample rows to simulate reduced sequencing depth
            if self.subsample_fraction < 1.0:
                df = df.sample(frac=self.subsample_fraction, random_state=self.subsample_seed)

            if use_cache:
                self._repertoire_cache[file_path] = df
            
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
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
    
    def extract_4mer_features(self, file_path, use_cache=True):
        """
        Extract 4-mer instances from a repertoire file.
        
        Paper excludes first 3 and last 3 residues of CDR3 sequences,
        then slides a window of 4 amino acids.
        
        Args:
            file_path: Path to the repertoire file
            use_cache: Whether to use cached features (default: True)
            
        Returns:
            Tuple of (feature_matrix, abundance_features) where:
            - feature_matrix: (N, 20) array of Atchley-encoded 4-mers
            - abundance_features: (N,) array of log abundances
        """
        if use_cache and file_path in self._features_cache:
            return self._features_cache[file_path]
        
        df = self.load_repertoire(file_path)
        
        if df.empty:
            return np.array([]).reshape(0, 20), np.array([])
        
        # Calculate total templates for normalization
        if 'templates' not in df.columns:
            df = df.copy()
            df['templates'] = 1
        
        total_count = df['templates'].sum()
        
        all_features = []
        all_abundances = []
        
        for _, row in df.iterrows():
            cdr3 = row[self.sequence_col]
            
            if not isinstance(cdr3, str) or len(cdr3) < self.min_cdr3_length:
                continue
            
            count = row['templates'] if 'templates' in row else 1
            
            # Extract core sequence (cut off first 3 and last 3)
            core_sequence = cdr3[3:-3]
            
            # Slide window to get 4-mers
            for i in range(len(core_sequence) - 3):
                four_mer = core_sequence[i:i+4]
                
                # Check for invalid characters
                if any(aa not in self.atchley for aa in four_mer):
                    continue
                
                # Encode 4-mer using Atchley factors (4 residues * 5 factors = 20 features)
                features = []
                for aa in four_mer:
                    features.extend(self.atchley[aa])
                
                # Calculate log relative abundance
                rel_abundance = count / total_count if total_count > 0 else 0
                log_abundance = np.log(rel_abundance) if rel_abundance > 0 else -10.0
                
                all_features.append(features)
                all_abundances.append(log_abundance)
        
        if not all_features:
            result = (np.array([]).reshape(0, 20), np.array([]))
        else:
            result = (np.array(all_features), np.array(all_abundances))
        
        if use_cache:
            self._features_cache[file_path] = result
        
        return result
    
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
            return np.array([0.5])  # Default to 0.5 if no valid instances
        
        # Combine features: 20 Atchley + 1 abundance = 21 features
        combined_features = np.column_stack([features, abundances])
        
        # Compute logits: X @ weights + bias
        logits = combined_features @ self.weights + self.bias
        
        # Apply sigmoid
        return self._sigmoid(logits)
    
    def _compute_bag_probability(self, instance_scores):
        """
        MIL aggregation using max pooling.
        
        The bag (repertoire) probability is the maximum instance probability.
        
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
        Compute the loss and gradient for the MIL model.
        
        Uses cross-entropy loss with L2 regularization.
        Gradient is computed using the chain rule through max pooling.
        
        Args:
            params: Flattened parameters (weights + bias)
            training_data: List of (features, abundances) tuples
            labels: List of labels
            
        Returns:
            Tuple of (loss, gradient)
        """
        self.weights = params[:21]
        self.bias = params[21]
        
        total_loss = 0.0
        total_grad = np.zeros(22)
        
        for (features, abundances), label in zip(training_data, labels):
            if len(features) == 0:
                continue
            
            # Forward pass
            combined_features = np.column_stack([features, abundances])
            logits = combined_features @ self.weights + self.bias
            instance_probs = self._sigmoid(logits)
            
            # MIL: Find max instance (the "witness")
            max_idx = np.argmax(instance_probs)
            bag_prob = instance_probs[max_idx]
            
            # Compute cross-entropy loss
            eps = 1e-10
            bag_prob_clipped = np.clip(bag_prob, eps, 1 - eps)
            loss = -(label * np.log(bag_prob_clipped) + 
                     (1 - label) * np.log(1 - bag_prob_clipped))
            total_loss += loss
            
            # Backpropagation through max (only the witness contributes)
            # d_loss/d_bag_prob
            d_loss_d_prob = (bag_prob_clipped - label) / (bag_prob_clipped * (1 - bag_prob_clipped))
            
            # d_bag_prob/d_logit (sigmoid derivative)
            d_prob_d_logit = bag_prob_clipped * (1 - bag_prob_clipped)
            
            # Combined
            d_loss_d_logit = d_loss_d_prob * d_prob_d_logit
            
            # d_logit/d_weights and d_logit/d_bias
            witness_features = combined_features[max_idx]
            d_loss_d_weights = d_loss_d_logit * witness_features
            d_loss_d_bias = d_loss_d_logit
            
            total_grad[:21] += d_loss_d_weights
            total_grad[21] += d_loss_d_bias
        
        # Average over samples
        n_samples = len(labels)
        total_loss /= n_samples
        total_grad /= n_samples
        
        # L2 regularization (don't regularize bias)
        total_loss += 0.5 * self.reg_strength * np.sum(self.weights ** 2)
        total_grad[:21] += self.reg_strength * self.weights
        
        return total_loss, total_grad
    
    def train(self, training_files, training_labels):
        """
        Train the MIL-TCR model.
        
        Args:
            training_files: List of file paths to patient .tsv files
            training_labels: List of integers (1 for Diseased, 0 for Healthy)
            
        Returns:
            Dictionary with training results
        """
        print("--- Training MIL-TCR Classifier ---")
        
        # Extract features for all training samples
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
        
        print(f"Training on {len(training_data)} samples...")
        
        # Initialize parameters
        initial_params = np.zeros(22)  # 21 weights + 1 bias
        initial_params[:21] = np.random.randn(21) * 0.01
        
        # Optimize using L-BFGS-B
        def objective(params):
            loss, grad = self._compute_loss_and_gradient(params, training_data, valid_labels)
            return loss, grad
        
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.max_iter, 'disp': False}
        )
        
        self.weights = result.x[:21]
        self.bias = result.x[21]
        
        print(f"Training completed. Final loss: {result.fun:.4f}")
        
        # Compute training accuracy
        train_preds = []
        for features, abundances in training_data:
            instance_scores = self._compute_instance_scores(features, abundances)
            bag_prob = self._compute_bag_probability(instance_scores)
            train_preds.append(1 if bag_prob >= 0.5 else 0)
        
        train_accuracy = np.mean(np.array(train_preds) == np.array(valid_labels))
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return {
            'final_loss': result.fun,
            'train_accuracy': train_accuracy,
            'n_samples': len(training_data),
            'converged': result.success
        }
    
    def predict_diagnosis(self, file_path):
        """
        Predict disease status for a new patient.
        
        Args:
            file_path: Path to the patient's repertoire TSV file
            
        Returns:
            Dictionary with prediction results including:
            - n_instances: Number of 4-mer instances extracted
            - probability_positive: Probability of disease
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
        Get the most predictive 4-mer motifs from a repertoire.
        
        Args:
            file_path: Path to the patient's repertoire TSV file
            threshold: Score threshold for "diagnostic" motifs (default: 0.5)
            top_k: Number of top motifs to return (default: 10)
            
        Returns:
            DataFrame with top diagnostic motifs
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df = self.load_repertoire(file_path)
        
        if df.empty:
            return pd.DataFrame()
        
        if 'templates' not in df.columns:
            df = df.copy()
            df['templates'] = 1
        
        total_count = df['templates'].sum()
        
        motifs = []
        
        for _, row in df.iterrows():
            cdr3 = row[self.sequence_col]
            
            if not isinstance(cdr3, str) or len(cdr3) < self.min_cdr3_length:
                continue
            
            count = row['templates'] if 'templates' in row else 1
            core_sequence = cdr3[3:-3]
            
            for i in range(len(core_sequence) - 3):
                four_mer = core_sequence[i:i+4]
                
                if any(aa not in self.atchley for aa in four_mer):
                    continue
                
                features = []
                for aa in four_mer:
                    features.extend(self.atchley[aa])
                
                rel_abundance = count / total_count if total_count > 0 else 0
                log_abundance = np.log(rel_abundance) if rel_abundance > 0 else -10.0
                
                combined = np.concatenate([features, [log_abundance]])
                logit = np.dot(self.weights, combined) + self.bias
                score = self._sigmoid(logit)
                
                motifs.append({
                    '4mer': four_mer,
                    'parent_cdr3': cdr3,
                    'score': float(score),
                    'log_abundance': log_abundance
                })
        
        if not motifs:
            return pd.DataFrame()
        
        motifs_df = pd.DataFrame(motifs)
        motifs_df = motifs_df.sort_values('score', ascending=False)
        
        # Filter by threshold and return top_k
        diagnostic = motifs_df[motifs_df['score'] >= threshold].head(top_k)
        
        return diagnostic


# --- Usage Example ---

if __name__ == "__main__":
    print("Ostmeyer 2019 MIL-TCR Classification Model")
    print("=" * 60)
    print("\nThis module contains the core model implementation.")
    print("For evaluation with cross-validation, use:")
    print("  from evals.ostmeyer_2019_disease_classification import Ostmeyer2019Evaluator")
    print("\nBasic usage example:")
    print("  model = MIL_TCR_Classifier(learning_rate=0.01, max_iter=100)")
    print("  model.train(train_files, train_labels)")
    print("  result = model.predict_diagnosis('patient.tsv.gz')")
    
    # Example (commented out as files don't exist):
    # train_files = ['patient1.tsv.gz', 'patient2.tsv.gz', ...]
    # train_labels = [1, 0, ...] 
    
    # model = MIL_TCR_Classifier()
    # model.train(train_files, train_labels)
    # result = model.predict_diagnosis('new_patient.tsv.gz')
    # print(result)
