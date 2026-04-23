"""
XGBoost classifier on gapped k-mer + V/J gene features.

Mirrors the ensemble structure of Gapped_4mer_VJgene (ensemble_regression.py):
- Separate XGBoost trained on k-mer features
- Separate XGBoost trained on V/J gene features
- Linear weighted ensemble: alpha * kmer_prob + (1-alpha) * vj_prob
  (alpha tuned via sweep on an internal 80/20 validation split)

Hyperparameters for each sub-model tuned via a small CV grid over
max_depth × learning_rate with early stopping — no Optuna.
"""

import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import xgboost as xgb

from models.ensemble_regression import _extract_kmers
from utils.repertoire_io import load_raw_repertoire


class XGBoostKmer:
    """
    Ensemble of two XGBoost models: one on gapped k-mers, one on V/J gene counts.
    Combined with a linear alpha sweep on a held-out validation split.
    Supports 'ensemble', 'kmer_only', and 'vj_only' submodels.
    """

    MAX_DEPTH_CANDIDATES = [3, 5, 7]
    LEARNING_RATE_CANDIDATES = [0.05, 0.1, 0.2]
    VALID_SUBMODELS = ('ensemble', 'kmer_only', 'vj_only')

    def __init__(self, val_split=0.2, n_cv_folds=3, sequence_col='cdr3_aa',
                 v_gene_col='v_call', j_gene_col='j_call',
                 subsample_fraction=1.0, subsample_seed=7, subsample_n=None,
                 indices_map=None, ignore_allele=False,
                 early_stopping_rounds=20, n_jobs=None,
                 kmer_size=4, use_gaps=True, submodel='ensemble'):
        if submodel not in self.VALID_SUBMODELS:
            raise ValueError(f"submodel must be one of {self.VALID_SUBMODELS}")
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
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.kmer_size = kmer_size
        self.use_gaps = use_gaps
        self.submodel = submodel

        self._repertoire_cache = {}
        self._kmer_features_cache = {}
        self._vj_features_cache = {}

        self.kmer_vectorizer = None
        self.vj_vectorizer = None
        self.kmer_model = None
        self.vj_model = None
        self.best_alpha = 0.5

    # ------------------------------------------------------------------
    # Repertoire loading and feature extraction
    # ------------------------------------------------------------------

    def load_repertoire(self, file_path, use_cache=True):
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
        for fp in tqdm(file_paths, desc="Preloading repertoires"):
            self.load_repertoire(fp, use_cache=True)

    def _normalize_gene(self, gene):
        if self.ignore_allele and isinstance(gene, str):
            gene = gene.split('*')[0]
        return gene

    def _get_kmer_feature_dict(self, file_path):
        if file_path in self._kmer_features_cache:
            return self._kmer_features_cache[file_path]
        df = self.load_repertoire(file_path)
        counts = {}
        for seq in df[self.sequence_col].dropna():
            for kmer in _extract_kmers(str(seq), self.kmer_size, self.use_gaps):
                counts[kmer] = counts.get(kmer, 0) + 1
        self._kmer_features_cache[file_path] = counts
        return counts

    def _get_vj_feature_dict(self, file_path):
        if file_path in self._vj_features_cache:
            return self._vj_features_cache[file_path]
        df = self.load_repertoire(file_path)
        counts = {}
        if self.v_gene_col in df.columns:
            for gene in df[self.v_gene_col].dropna():
                key = f'V:{self._normalize_gene(gene)}'
                counts[key] = counts.get(key, 0) + 1
        if self.j_gene_col in df.columns:
            for gene in df[self.j_gene_col].dropna():
                key = f'J:{self._normalize_gene(gene)}'
                counts[key] = counts.get(key, 0) + 1
        self._vj_features_cache[file_path] = counts
        return counts

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def _tune_xgb_params(self, X, y, label):
        """Grid search over max_depth × learning_rate via stratified k-fold CV."""
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True,
                              random_state=self.subsample_seed)
        best_params = {'max_depth': 5, 'learning_rate': 0.1}
        best_auroc = -1.0

        for max_depth in self.MAX_DEPTH_CANDIDATES:
            for lr in self.LEARNING_RATE_CANDIDATES:
                fold_aurocs = []
                for tr_idx, vl_idx in skf.split(X, y):
                    if len(np.unique(y[vl_idx])) < 2:
                        continue
                    dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
                    dval = xgb.DMatrix(X[vl_idx], label=y[vl_idx])
                    params = self._base_params(max_depth, lr)
                    bst = xgb.train(
                        params, dtrain,
                        num_boost_round=500,
                        evals=[(dval, 'val')],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=False,
                    )
                    fold_aurocs.append(roc_auc_score(y[vl_idx], bst.predict(dval)))

                if fold_aurocs and np.mean(fold_aurocs) > best_auroc:
                    best_auroc = np.mean(fold_aurocs)
                    best_params = {'max_depth': max_depth, 'learning_rate': lr}

        print(f"  Best {label} params: {best_params}, CV AUROC: {best_auroc:.4f}")
        return best_params

    def _base_params(self, max_depth, lr):
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': max_depth,
            'learning_rate': lr,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'nthread': self.n_jobs or 0,
            'verbosity': 0,
        }

    def _train_submodel(self, X_base, y_base, X_val, y_val, params):
        """Train a single XGBoost with early stopping on the val split."""
        dtrain = xgb.DMatrix(X_base, label=y_base)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        return model

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def train(self, train_files, train_labels):
        train_files = list(train_files)
        train_labels = np.array(train_labels)

        use_kmer = self.submodel in ('ensemble', 'kmer_only')
        use_vj = self.submodel in ('ensemble', 'vj_only')

        self.preload_repertoires(train_files)

        base_idx, val_idx = train_test_split(
            np.arange(len(train_files)), test_size=self.val_split,
            random_state=self.subsample_seed, stratify=train_labels,
        )
        y_base = train_labels[base_idx]
        y_val = train_labels[val_idx]

        best_params_kmer = best_params_vj = None
        kmer_val_probs = vj_val_probs = None

        if use_kmer:
            print("Extracting k-mer features...")
            kmer_dicts = [self._get_kmer_feature_dict(f) for f in tqdm(train_files, leave=False)]
            self.kmer_vectorizer = DictVectorizer(sparse=True)
            X_kmer = self.kmer_vectorizer.fit_transform(kmer_dicts)
            X_kmer_base, X_kmer_val = X_kmer[base_idx], X_kmer[val_idx]

            print("Tuning k-mer XGBoost...")
            best_params_kmer = self._tune_xgb_params(X_kmer_base, y_base, 'k-mer')
            self.kmer_model = self._train_submodel(
                X_kmer_base, y_base, X_kmer_val, y_val,
                self._base_params(best_params_kmer['max_depth'],
                                  best_params_kmer['learning_rate']),
            )
            kmer_val_probs = self.kmer_model.predict(
                xgb.DMatrix(X_kmer_val),
                iteration_range=(0, self.kmer_model.best_ntree_limit),
            )
            print(f"  k-mer model: {self.kmer_model.best_ntree_limit} trees, "
                  f"val AUROC: {roc_auc_score(y_val, kmer_val_probs):.4f}")

        if use_vj:
            print("Extracting V/J gene features...")
            vj_dicts = [self._get_vj_feature_dict(f) for f in tqdm(train_files, leave=False)]
            self.vj_vectorizer = DictVectorizer(sparse=True)
            X_vj = self.vj_vectorizer.fit_transform(vj_dicts)
            X_vj_base, X_vj_val = X_vj[base_idx], X_vj[val_idx]

            print("Tuning V/J XGBoost...")
            best_params_vj = self._tune_xgb_params(X_vj_base, y_base, 'V/J')
            self.vj_model = self._train_submodel(
                X_vj_base, y_base, X_vj_val, y_val,
                self._base_params(best_params_vj['max_depth'],
                                  best_params_vj['learning_rate']),
            )
            vj_val_probs = self.vj_model.predict(
                xgb.DMatrix(X_vj_val),
                iteration_range=(0, self.vj_model.best_ntree_limit),
            )
            print(f"  V/J model: {self.vj_model.best_ntree_limit} trees, "
                  f"val AUROC: {roc_auc_score(y_val, vj_val_probs):.4f}")

        # Alpha sweep on val split
        best_val_auroc = -1.0
        if self.submodel == 'ensemble' and len(np.unique(y_val)) >= 2:
            print("Tuning ensemble alpha...")
            for alpha in np.arange(0.0, 1.01, 0.1):
                ensemble_probs = alpha * kmer_val_probs + (1 - alpha) * vj_val_probs
                auroc = roc_auc_score(y_val, ensemble_probs)
                if auroc > best_val_auroc:
                    best_val_auroc = auroc
                    self.best_alpha = alpha
            print(f"  Best alpha (k-mer weight): {self.best_alpha:.1f}, "
                  f"val AUROC: {best_val_auroc:.4f}")
        elif self.submodel == 'kmer_only':
            self.best_alpha = 1.0
            best_val_auroc = roc_auc_score(y_val, kmer_val_probs)
        elif self.submodel == 'vj_only':
            self.best_alpha = 0.0
            best_val_auroc = roc_auc_score(y_val, vj_val_probs)

        return {
            'best_params_kmer': best_params_kmer,
            'best_params_vj': best_params_vj,
            'best_alpha': self.best_alpha,
            'val_auroc': best_val_auroc,
        }

    def predict_diagnosis(self, file_path):
        kmer_prob = vj_prob = None

        if self.submodel in ('ensemble', 'kmer_only'):
            X_kmer = self.kmer_vectorizer.transform([self._get_kmer_feature_dict(file_path)])
            kmer_prob = float(self.kmer_model.predict(
                xgb.DMatrix(X_kmer),
                iteration_range=(0, self.kmer_model.best_ntree_limit),
            )[0])

        if self.submodel in ('ensemble', 'vj_only'):
            X_vj = self.vj_vectorizer.transform([self._get_vj_feature_dict(file_path)])
            vj_prob = float(self.vj_model.predict(
                xgb.DMatrix(X_vj),
                iteration_range=(0, self.vj_model.best_ntree_limit),
            )[0])

        if self.submodel == 'ensemble':
            prob = self.best_alpha * kmer_prob + (1 - self.best_alpha) * vj_prob
        elif self.submodel == 'kmer_only':
            prob = kmer_prob
        else:
            prob = vj_prob

        return {
            'probability_positive': prob,
            'kmer_probability': kmer_prob,
            'vj_probability': vj_prob,
            'best_alpha': self.best_alpha,
            'diagnosis': 'Diseased' if prob >= 0.5 else 'Healthy',
        }
