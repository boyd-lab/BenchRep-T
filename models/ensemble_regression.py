"""
Ensemble Regression model: Gapped 4-mer + V/J gene logistic regression ensemble.

Implements the method from the AIRR-ML Kaggle competition:
- 4-mer features with all single-position gapped variants from CDR3 sequences
- V gene and J gene frequency features
- L1 logistic regression base models (C tuned via 5-fold CV by AUROC)
- Linear weighted ensemble: alpha * kmer_prob + (1-alpha) * vj_prob
  (alpha tuned via sweep on an internal 80/20 validation split)
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.repertoire_io import load_raw_repertoire


def _extract_kmers(sequence, k, use_gaps=True):
    """
    Extract k-mers (and optionally their single-gap variants) from a sequence.

    For each k-residue window, yields the k-mer and, when use_gaps=True,
    k additional variants with one position replaced by '_'.
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        window = sequence[i:i + k]
        kmers.append(window)
        if use_gaps:
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

    VALID_SUBMODELS = ('ensemble', 'kmer_only', 'vj_only')

    def __init__(self, val_split=0.2, n_cv_folds=5, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None,
                 ignore_allele=False,
                 canonicalize_genes=False,
                 submodel='ensemble',
                 kmer_size=4,
                 use_gaps=True,
                 n_jobs=None):
        """
        Args:
            val_split: Fraction of training data held out internally for alpha tuning.
            n_cv_folds: Number of CV folds used for C hyperparameter tuning.
            sequence_col: Column containing CDR3 amino acid sequences.
            v_gene_col: Column containing V gene calls.
            j_gene_col: Column containing J gene calls (skipped if absent).
            subsample_fraction: Fraction of reads to sample per repertoire (depth sim).
            subsample_seed: Random seed for reproducibility.
            subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set).
            indices_map: Dict mapping rep_id to pre-computed row indices (default: None).
            ignore_allele: If True, strip allele designations (*XX) from V/J gene
                           names, enabling gene-level feature matching across datasets.
            canonicalize_genes: If True, additionally collapse Adaptive-style "-1"
                           suffixes on IMGT singleton TRBV families (e.g.
                           TRBV13-1 → TRBV13). Required when mixing internal IMGT
                           and external Adaptive-derived repertoires in one model.
            submodel: Which sub-model(s) to use: 'ensemble' (default, both
                      combined), 'kmer_only' (gapped 4-mer only), or 'vj_only'
                      (V/J gene count only).
            kmer_size: Length of k-mers to extract (default: 4).
            use_gaps: If True, include single-position gapped variants of each
                      k-mer. If False, extract plain k-mers only (default: True).
            n_jobs: Number of jobs to request for logistic-regression fits.
        """
        if submodel not in self.VALID_SUBMODELS:
            raise ValueError(f"Invalid submodel '{submodel}'. "
                             f"Choose from: {self.VALID_SUBMODELS}")
        self.submodel = submodel
        self.kmer_size = kmer_size
        self.use_gaps = use_gaps
        self.val_split = val_split
        self.n_cv_folds = n_cv_folds
        self.sequence_col = sequence_col
        self.v_gene_col = v_gene_col
        self.j_gene_col = j_gene_col
        self.subsample_fraction = subsample_fraction
        self.subsample_seed = subsample_seed
        self.subsample_n = subsample_n
        self.indices_map = indices_map
        self.ignore_allele = ignore_allele
        self.canonicalize_genes = canonicalize_genes
        self.n_jobs = n_jobs

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

        # Selected feature names (set after train()); used for covariate-adjusted
        # classification, which residualizes the per-sample values at these
        # L1-selected columns and refits with logistic regression.
        self.kmer_selected_names = []
        self.vj_selected_names = []

    # ------------------------------------------------------------------
    # Repertoire loading and caching
    # ------------------------------------------------------------------

    def load_repertoire(self, file_path, use_cache=True):
        """Load a repertoire file into a DataFrame (with optional caching)."""
        if use_cache and file_path in self._repertoire_cache:
            return self._repertoire_cache[file_path]

        indices = None
        if self.indices_map is not None:
            rep_id = os.path.basename(file_path).replace('.tsv.gz', '').replace('.tsv', '')
            indices = self.indices_map.get(rep_id)

        df = load_raw_repertoire(file_path, self.subsample_n, self.subsample_fraction,
                                 self.subsample_seed, subsample_indices=indices)

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
        Extract gapped 4-mer relative frequencies from a repertoire file.
        Counts are normalized by the total k-mer count in the repertoire so
        that values are comparable across repertoires of different depths.
        Returns a dict {kmer_string: frequency}.
        """
        if file_path in self._kmer_features_cache:
            return self._kmer_features_cache[file_path]

        df = self.load_repertoire(file_path)
        counts = {}
        for seq in df[self.sequence_col].dropna():
            for kmer in _extract_kmers(str(seq), self.kmer_size, self.use_gaps):
                counts[kmer] = counts.get(kmer, 0) + 1

        total = sum(counts.values())
        if total > 0:
            counts = {k: v / total for k, v in counts.items()}

        self._kmer_features_cache[file_path] = counts
        return counts

    def _normalize_gene(self, gene):
        """Apply optional allele stripping (and IMGT-singleton collapse) to a gene."""
        if not isinstance(gene, str):
            return gene
        if self.canonicalize_genes:
            from utils.gene_harmonization import canonicalize_gene
            return canonicalize_gene(gene)
        if self.ignore_allele:
            return gene.split('*')[0]
        return gene

    def _get_vj_feature_dict(self, file_path):
        """
        Extract V gene and J gene relative frequencies from a repertoire file.
        V and J are each normalized by their own total (independent
        distributions over gene calls), so that values are comparable across
        repertoires of different depths.
        Returns a dict {'V:<gene>': frequency, 'J:<gene>': frequency}.
        """
        if file_path in self._vj_features_cache:
            return self._vj_features_cache[file_path]

        df = self.load_repertoire(file_path)
        v_counts = {}
        j_counts = {}

        if self.v_gene_col in df.columns:
            for gene in df[self.v_gene_col].dropna():
                gene = self._normalize_gene(gene)
                v_counts[gene] = v_counts.get(gene, 0) + 1

        if self.j_gene_col in df.columns:
            for gene in df[self.j_gene_col].dropna():
                gene = self._normalize_gene(gene)
                j_counts[gene] = j_counts.get(gene, 0) + 1

        v_total = sum(v_counts.values())
        j_total = sum(j_counts.values())

        counts = {}
        if v_total > 0:
            for gene, c in v_counts.items():
                counts[f'V:{gene}'] = c / v_total
        if j_total > 0:
            for gene, c in j_counts.items():
                counts[f'J:{gene}'] = c / j_total

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
                                         max_iter=1000, n_jobs=self.n_jobs)
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
        Train the model according to self.submodel.

        For 'ensemble': trains both sub-models and tunes alpha on a val split.
        For 'kmer_only': trains only the gapped 4-mer sub-model.
        For 'vj_only': trains only the V/J gene count sub-model.

        Args:
            train_files: List of repertoire file paths.
            train_labels: List/array of binary labels (0 = healthy, 1 = disease).

        Returns:
            Dict with training summary.
        """
        train_files = list(train_files)
        train_labels = np.array(train_labels)
        use_kmer = self.submodel in ('ensemble', 'kmer_only')
        use_vj = self.submodel in ('ensemble', 'vj_only')

        self.preload_repertoires(train_files)

        # --- Feature extraction ---
        best_c_kmer = None
        best_c_vj = None
        n_kmer_features = 0
        n_vj_features = 0

        if use_kmer:
            print("Extracting k-mer features...")
            kmer_dicts = [self._get_kmer_feature_dict(f) for f in tqdm(train_files, leave=False)]
            self.kmer_vectorizer = DictVectorizer(sparse=True)
            X_kmer_all = self.kmer_vectorizer.fit_transform(kmer_dicts)
            n_kmer_features = X_kmer_all.shape[1]

        if use_vj:
            print("Extracting V/J gene features...")
            vj_dicts = [self._get_vj_feature_dict(f) for f in tqdm(train_files, leave=False)]
            self.vj_vectorizer = DictVectorizer(sparse=True)
            X_vj_all = self.vj_vectorizer.fit_transform(vj_dicts)
            n_vj_features = X_vj_all.shape[1]

        # --- Internal train/val split ---
        indices = np.arange(len(train_files))
        base_idx, val_idx = train_test_split(
            indices, test_size=self.val_split,
            random_state=self.subsample_seed,
            stratify=train_labels
        )
        y_base = train_labels[base_idx]
        y_val = train_labels[val_idx]

        # --- Tune and train k-mer model ---
        if use_kmer:
            X_kmer_base = X_kmer_all[base_idx]
            X_kmer_val = X_kmer_all[val_idx]

            print("Tuning k-mer model C...")
            best_c_kmer = self._tune_c(X_kmer_base, y_base, self.C_KMER_CANDIDATES)
            print(f"  Best C (k-mer): {best_c_kmer}")

            self.kmer_scaler = StandardScaler(with_mean=False)
            X_kmer_base_sc = self.kmer_scaler.fit_transform(X_kmer_base)
            self.kmer_model = LogisticRegression(C=best_c_kmer, penalty='l1',
                                                  solver='liblinear', max_iter=1000,
                                                  n_jobs=self.n_jobs)
            self.kmer_model.fit(X_kmer_base_sc, y_base)
            kmer_nonzero = np.flatnonzero(self.kmer_model.coef_.ravel())
            kmer_feature_names = self.kmer_vectorizer.get_feature_names_out()
            self.kmer_selected_names = [str(kmer_feature_names[i]) for i in kmer_nonzero]

        # --- Tune and train V/J model ---
        if use_vj:
            X_vj_base = X_vj_all[base_idx]
            X_vj_val = X_vj_all[val_idx]

            print("Tuning V/J model C...")
            best_c_vj = self._tune_c(X_vj_base, y_base, self.C_VJ_CANDIDATES)
            print(f"  Best C (V/J):   {best_c_vj}")

            self.vj_scaler = StandardScaler(with_mean=False)
            X_vj_base_sc = self.vj_scaler.fit_transform(X_vj_base)
            self.vj_model = LogisticRegression(C=best_c_vj, penalty='l1',
                                                solver='liblinear', max_iter=1000,
                                                n_jobs=self.n_jobs)
            self.vj_model.fit(X_vj_base_sc, y_base)
            vj_nonzero = np.flatnonzero(self.vj_model.coef_.ravel())
            vj_feature_names = self.vj_vectorizer.get_feature_names_out()
            self.vj_selected_names = [str(vj_feature_names[i]) for i in vj_nonzero]

        # --- Alpha sweep (ensemble only) ---
        best_alpha = 0.5
        best_val_auroc = -1.0

        if self.submodel == 'ensemble':
            print("Tuning ensemble alpha...")
            X_kmer_val_sc = self.kmer_scaler.transform(X_kmer_val)
            X_vj_val_sc = self.vj_scaler.transform(X_vj_val)

            kmer_val_probs = self.kmer_model.predict_proba(X_kmer_val_sc)[:, 1]
            vj_val_probs = self.vj_model.predict_proba(X_vj_val_sc)[:, 1]

            if len(np.unique(y_val)) >= 2:
                for alpha in np.arange(0.0, 1.01, 0.1):
                    ensemble_probs = alpha * kmer_val_probs + (1 - alpha) * vj_val_probs
                    auroc = roc_auc_score(y_val, ensemble_probs)
                    if auroc > best_val_auroc:
                        best_val_auroc = auroc
                        best_alpha = alpha
            else:
                print("  Warning: val set has only one class; using default alpha=0.5")

            print(f"  Best alpha (k-mer weight): {best_alpha:.1f}, "
                  f"Val AUROC: {best_val_auroc:.4f}")
        elif self.submodel == 'kmer_only':
            best_alpha = 1.0
            X_kmer_val_sc = self.kmer_scaler.transform(X_kmer_val)
            kmer_val_probs = self.kmer_model.predict_proba(X_kmer_val_sc)[:, 1]
            if len(np.unique(y_val)) >= 2:
                best_val_auroc = roc_auc_score(y_val, kmer_val_probs)
            print(f"  K-mer only Val AUROC: {best_val_auroc:.4f}")
        elif self.submodel == 'vj_only':
            best_alpha = 0.0
            X_vj_val_sc = self.vj_scaler.transform(X_vj_val)
            vj_val_probs = self.vj_model.predict_proba(X_vj_val_sc)[:, 1]
            if len(np.unique(y_val)) >= 2:
                best_val_auroc = roc_auc_score(y_val, vj_val_probs)
            print(f"  V/J only Val AUROC: {best_val_auroc:.4f}")

        self.best_alpha = best_alpha

        return {
            'best_c_kmer': best_c_kmer,
            'best_c_vj': best_c_vj,
            'best_alpha': best_alpha,
            'val_auroc': best_val_auroc,
            'n_kmer_features': n_kmer_features,
            'n_vj_features': n_vj_features,
        }

    def get_selected_features(self, file_path):
        """
        Return a per-sample dense vector of L1-selected feature values.

        Concatenates the per-repertoire normalized values at the columns where
        the trained k-mer and V/J L1 logistic regressions retained non-zero
        coefficients (subject to ``self.submodel``). Used for covariate-adjusted
        classification, where the resulting (n_samples, n_selected) matrix is
        residualized against demographics before refitting an L1 logistic
        regression.

        Returns:
            np.ndarray of shape (n_selected_kmer + n_selected_vj,) with dtype
            float32. Empty array if no features were selected.
        """
        vecs = []
        if self.submodel in ('ensemble', 'kmer_only') and self.kmer_selected_names:
            kmer_dict = self._get_kmer_feature_dict(file_path)
            vecs.append(np.array(
                [kmer_dict.get(name, 0) for name in self.kmer_selected_names],
                dtype=np.float32,
            ))
        if self.submodel in ('ensemble', 'vj_only') and self.vj_selected_names:
            vj_dict = self._get_vj_feature_dict(file_path)
            vecs.append(np.array(
                [vj_dict.get(name, 0) for name in self.vj_selected_names],
                dtype=np.float32,
            ))
        if not vecs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(vecs)

    @property
    def n_selected_features(self):
        """Total number of L1-selected features across active sub-models."""
        n = 0
        if self.submodel in ('ensemble', 'kmer_only'):
            n += len(self.kmer_selected_names)
        if self.submodel in ('ensemble', 'vj_only'):
            n += len(self.vj_selected_names)
        return n

    def predict_diagnosis(self, file_path):
        """
        Predict disease probability for a single repertoire file.

        Args:
            file_path: Path to a repertoire .tsv / .tsv.gz file.

        Returns:
            Dict with:
              probability_positive (float): P(disease) in [0, 1]
              kmer_probability (float): k-mer model P(disease) (None if vj_only)
              vj_probability (float): V/J model P(disease) (None if kmer_only)
              best_alpha (float): ensemble weight used for k-mer model
              diagnosis (str): 'Diseased' or 'Healthy'
        """
        kmer_prob = None
        vj_prob = None

        if self.submodel in ('ensemble', 'kmer_only'):
            kmer_dict = self._get_kmer_feature_dict(file_path)
            X_kmer = self.kmer_scaler.transform(self.kmer_vectorizer.transform([kmer_dict]))
            kmer_prob = float(self.kmer_model.predict_proba(X_kmer)[0, 1])

        if self.submodel in ('ensemble', 'vj_only'):
            vj_dict = self._get_vj_feature_dict(file_path)
            X_vj = self.vj_scaler.transform(self.vj_vectorizer.transform([vj_dict]))
            vj_prob = float(self.vj_model.predict_proba(X_vj)[0, 1])

        if self.submodel == 'ensemble':
            prob = self.best_alpha * kmer_prob + (1 - self.best_alpha) * vj_prob
        elif self.submodel == 'kmer_only':
            prob = kmer_prob
        else:  # vj_only
            prob = vj_prob

        return {
            'probability_positive': prob,
            'kmer_probability': kmer_prob,
            'vj_probability': vj_prob,
            'best_alpha': self.best_alpha,
            'diagnosis': 'Diseased' if prob >= 0.5 else 'Healthy',
        }
