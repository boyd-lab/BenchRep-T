"""
ML Baseline model: Gapped 4-mer + V/J gene logistic regression ensemble.

Implements the method from the AIRR-ML Kaggle competition:
- 4-mer features with all single-position gapped variants from CDR3 sequences
- V gene and J gene frequency features
- L1 logistic regression base models (C tuned via 5-fold CV by AUROC)
- Linear weighted ensemble: alpha * kmer_prob + (1-alpha) * vj_prob
  (alpha tuned via sweep on an internal 80/20 validation split)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def _extract_gapped_4mers(sequence):
    """
    Extract 4-mers and their single-gap variants from a sequence.

    For each 4-residue window, yields:
      - the original 4-mer
      - 4 gapped variants (one position replaced by '_' per variant)

    Returns a list of k-mer strings.
    """
    k = 4
    kmers = []
    for i in range(len(sequence) - k + 1):
        window = sequence[i:i + k]
        kmers.append(window)
        for j in range(k):
            kmers.append(window[:j] + '_' + window[j + 1:])
    return kmers


class Gapped_4mer_VJgene:
    """
    Ensemble classifier combining:
      1. Gapped 4-mer logistic regression on CDR3 sequences
      2. V/J gene frequency logistic regression

    Uses a linear weighted average (alpha sweep on an internal validation split)
    as the meta-learner.
    """

    # Default C grids (from the original Kaggle solution)
    C_KMER_CANDIDATES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    C_VJ_CANDIDATES = [1.0, 0.2, 0.1, 0.05, 0.03]

    def __init__(self, val_split=0.2, n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7):
        """
        Args:
            val_split: Fraction of training data held out internally for alpha tuning.
            n_cv_folds: Number of CV folds used for C hyperparameter tuning.
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls (skipped if absent).
            subsample_fraction: Fraction of reads to sample per repertoire (depth sim).
            subsample_seed: Random seed for reproducibility.
        """
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed

        # Caches
        self._repertoire_cache = {}
        self._kmer_features_cache = {}
        self._vj_features_cache = {}

        # Trained components (set after train())
        self.kmer_vectorizer = None
        self.kmer_scaler = None
        self.kmer_model = None
        self.vj_vectorizer = None
        self.vj_scaler = None
        self.vj_model = None
        self.best_alpha = 0.5  # weight for kmer model in ensemble

    # ------------------------------------------------------------------
    # Repertoire loading and caching
    # ------------------------------------------------------------------

    def load_repertoire(self, file_path, use_cache=True):
        """Load a repertoire file into a DataFrame (with optional caching)."""
        if use_cache and file_path in self._repertoire_cache:
            return self._repertoire_cache[file_path]

        df = pd.read_csv(file_path, sep='\t', compression='infer', low_memory=False)

        if self.subsample_fraction < 1.0:
            rng = np.random.default_rng(self.subsample_seed)
            n = max(1, int(len(df) * self.subsample_fraction))
            df = df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(drop=True)

        if use_cache:
            self._repertoire_cache[file_path] = df
        return df

    def preload_repertoires(self, file_paths):
        """Pre-load multiple repertoire files into cache."""
        for fp in tqdm(file_paths, desc="Preloading repertoires"):
            self.load_repertoire(fp, use_cache=True)

    def clear_cache(self):
        """Clear all cached data."""
        self._repertoire_cache.clear()
        self._kmer_features_cache.clear()
        self._vj_features_cache.clear()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_kmer_feature_dict(self, file_path):
        """
        Extract gapped 4-mer counts from a repertoire file.
        Returns a dict {kmer_string: count}.
        """
        if file_path in self._kmer_features_cache:
            return self._kmer_features_cache[file_path]

        df = self.load_repertoire(file_path)
        counts = {}
        for seq in df[self.sequence_col].dropna():
            for kmer in _extract_gapped_4mers(str(seq)):
                counts[kmer] = counts.get(kmer, 0) + 1

        self._kmer_features_cache[file_path] = counts
        return counts

    def _get_vj_feature_dict(self, file_path):
        """
        Extract V gene and J gene frequency counts from a repertoire file.
        Returns a dict {'V:<gene>': count, 'J:<gene>': count}.
        """
        if file_path in self._vj_features_cache:
            return self._vj_features_cache[file_path]

        df = self.load_repertoire(file_path)
        counts = {}

        if self.v_gene_col in df.columns:
            for gene in df[self.v_gene_col].dropna():
                key = f'V:{gene}'
                counts[key] = counts.get(key, 0) + 1

        if self.j_gene_col in df.columns:
            for gene in df[self.j_gene_col].dropna():
                key = f'J:{gene}'
                counts[key] = counts.get(key, 0) + 1

        self._vj_features_cache[file_path] = counts
        return counts

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _tune_c(self, X, y, c_candidates):
        """
        Select the best C for an L1 logistic regression via stratified k-fold CV.
        Returns the C value with the highest mean AUROC.
        """
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True,
                              random_state=self.subsample_seed)
        best_c, best_auroc = c_candidates[0], -1.0

        for c in c_candidates:
            fold_aurocs = []
            for tr_idx, vl_idx in skf.split(X, y):
                scaler = StandardScaler(with_mean=False)
                X_tr = scaler.fit_transform(X[tr_idx])
                X_vl = scaler.transform(X[vl_idx])
                clf = LogisticRegression(C=c, penalty='l1', solver='liblinear',
                                         max_iter=1000)
                clf.fit(X_tr, y[tr_idx])
                probs = clf.predict_proba(X_vl)[:, 1]
                # Skip fold if only one class present
                if len(np.unique(y[vl_idx])) < 2:
                    continue
                fold_aurocs.append(roc_auc_score(y[vl_idx], probs))

            if fold_aurocs:
                mean_auroc = np.mean(fold_aurocs)
                if mean_auroc > best_auroc:
                    best_auroc = mean_auroc
                    best_c = c

        return best_c

    # ------------------------------------------------------------------
    # Main training and prediction
    # ------------------------------------------------------------------

    def train(self, train_files, train_labels):
        """
        Train the ensemble model.

        Internally splits training data into base (80%) and val (20%):
          - Base set: C tuned via k-fold CV, base models trained.
          - Val set: alpha tuned via sweep (0.0 to 1.0, step 0.1).

        Args:
            train_files: List of repertoire file paths.
            train_labels: List/array of binary labels (0 = healthy, 1 = disease).

        Returns:
            Dict with training summary (best_c_kmer, best_c_vj, best_alpha, val_auroc).
        """
        train_files = list(train_files)
        train_labels = np.array(train_labels)

        self.preload_repertoires(train_files)

        # --- Feature extraction ---
        print("Extracting k-mer features...")
        kmer_dicts = [self._get_kmer_feature_dict(f) for f in tqdm(train_files, leave=False)]

        print("Extracting V/J gene features...")
        vj_dicts = [self._get_vj_feature_dict(f) for f in tqdm(train_files, leave=False)]

        # Fit vectorizers on the full training set so vocabulary covers all splits
        self.kmer_vectorizer = DictVectorizer(sparse=True)
        X_kmer_all = self.kmer_vectorizer.fit_transform(kmer_dicts)

        self.vj_vectorizer = DictVectorizer(sparse=True)
        X_vj_all = self.vj_vectorizer.fit_transform(vj_dicts)

        # --- Internal train/val split ---
        indices = np.arange(len(train_files))
        base_idx, val_idx = train_test_split(
            indices, test_size=self.val_split,
            random_state=self.subsample_seed,
            stratify=train_labels
        )

        X_kmer_base = X_kmer_all[base_idx]
        X_kmer_val = X_kmer_all[val_idx]
        X_vj_base = X_vj_all[base_idx]
        X_vj_val = X_vj_all[val_idx]
        y_base = train_labels[base_idx]
        y_val = train_labels[val_idx]

        # --- Tune and train k-mer model ---
        print("Tuning k-mer model C...")
        best_c_kmer = self._tune_c(X_kmer_base, y_base, self.C_KMER_CANDIDATES)
        print(f"  Best C (k-mer): {best_c_kmer}")

        self.kmer_scaler = StandardScaler(with_mean=False)
        X_kmer_base_sc = self.kmer_scaler.fit_transform(X_kmer_base)
        self.kmer_model = LogisticRegression(C=best_c_kmer, penalty='l1',
                                              solver='liblinear', max_iter=1000)
        self.kmer_model.fit(X_kmer_base_sc, y_base)

        # --- Tune and train V/J model ---
        print("Tuning V/J model C...")
        best_c_vj = self._tune_c(X_vj_base, y_base, self.C_VJ_CANDIDATES)
        print(f"  Best C (V/J):   {best_c_vj}")

        self.vj_scaler = StandardScaler(with_mean=False)
        X_vj_base_sc = self.vj_scaler.fit_transform(X_vj_base)
        self.vj_model = LogisticRegression(C=best_c_vj, penalty='l1',
                                            solver='liblinear', max_iter=1000)
        self.vj_model.fit(X_vj_base_sc, y_base)

        # --- Alpha sweep on validation set ---
        print("Tuning ensemble alpha...")
        X_kmer_val_sc = self.kmer_scaler.transform(X_kmer_val)
        X_vj_val_sc = self.vj_scaler.transform(X_vj_val)

        kmer_val_probs = self.kmer_model.predict_proba(X_kmer_val_sc)[:, 1]
        vj_val_probs = self.vj_model.predict_proba(X_vj_val_sc)[:, 1]

        best_alpha, best_val_auroc = 0.5, -1.0
        if len(np.unique(y_val)) >= 2:
            for alpha in np.arange(0.0, 1.01, 0.1):
                ensemble_probs = alpha * kmer_val_probs + (1 - alpha) * vj_val_probs
                auroc = roc_auc_score(y_val, ensemble_probs)
                if auroc > best_val_auroc:
                    best_val_auroc = auroc
                    best_alpha = alpha
        else:
            print("  Warning: val set has only one class; using default alpha=0.5")

        self.best_alpha = best_alpha
        print(f"  Best alpha (k-mer weight): {best_alpha:.1f}, "
              f"Val AUROC: {best_val_auroc:.4f}")

        return {
            'best_c_kmer': best_c_kmer,
            'best_c_vj': best_c_vj,
            'best_alpha': best_alpha,
            'val_auroc': best_val_auroc,
            'n_kmer_features': X_kmer_all.shape[1],
            'n_vj_features': X_vj_all.shape[1],
        }

    def predict_diagnosis(self, file_path):
        """
        Predict disease probability for a single repertoire file.

        Args:
            file_path: Path to a repertoire .tsv / .tsv.gz file.

        Returns:
            Dict with:
              probability_positive (float): ensemble P(disease) in [0, 1]
              kmer_probability (float): k-mer model P(disease)
              vj_probability (float): V/J model P(disease)
              best_alpha (float): ensemble weight used for k-mer model
              diagnosis (str): 'Diseased' or 'Healthy'
        """
        kmer_dict = self._get_kmer_feature_dict(file_path)
        vj_dict = self._get_vj_feature_dict(file_path)

        X_kmer = self.kmer_scaler.transform(self.kmer_vectorizer.transform([kmer_dict]))
        X_vj = self.vj_scaler.transform(self.vj_vectorizer.transform([vj_dict]))

        kmer_prob = float(self.kmer_model.predict_proba(X_kmer)[0, 1])
        vj_prob = float(self.vj_model.predict_proba(X_vj)[0, 1])

        prob = self.best_alpha * kmer_prob + (1 - self.best_alpha) * vj_prob

        return {
            'probability_positive': prob,
            'kmer_probability': kmer_prob,
            'vj_probability': vj_prob,
            'best_alpha': self.best_alpha,
            'diagnosis': 'Diseased' if prob >= 0.5 else 'Healthy',
        }
