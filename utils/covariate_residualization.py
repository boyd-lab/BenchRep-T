"""
Covariate residualization utility.

Fits an OLS regression predicting each feature from demographic covariates
(age, sex, ancestry) on the training fold, then returns the residuals as
de-confounded features.  Missing demographic values are encoded explicitly
rather than dropped:

  - age:      continuous; NaN imputed with training-set median.
  - sex:      two indicator columns (sex_M, sex_unknown); F-known is (0, 0).
  - ancestry: one-hot with a "Missing" category for NaN/blank values.

This means no samples are dropped due to incomplete demographics.

Typical usage (inside a CV loop):

    residualizer = CovariateResidualizer()
    X_train_res = residualizer.fit_transform(train_metadata, X_train)
    X_test_res  = residualizer.transform(test_metadata,  X_test)

Or use the high-level helper that handles everything in one call:

    test_probs = covariate_adjusted_predict(
        X_train, train_meta, y_train, X_test, test_meta
    )
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ANCESTRY_MISSING = "Missing"


def encode_covariates(metadata, ancestry_categories=None, age_median=None):
    """
    Encode demographic columns into a numeric matrix.

    Missing values are encoded explicitly rather than dropped:
      - age:      imputed with training mean + age_missing indicator column.
      - sex:      two columns — sex_M (1 if known-male) and sex_unknown
                  (1 if sex is NaN).  F-known → (0, 0).
      - ancestry: one-hot; NaN/blank values map to the "Missing" category.

    An intercept column (all ones) is prepended so the linear regression
    absorbs the per-feature mean, ensuring residuals are centred.

    Args:
        metadata: DataFrame with 'age', 'sex', and optionally 'ancestry'.
        ancestry_categories: Sorted list of ancestry strings including
            "Missing".  Pass the list from a training-data call so that test
            data uses the same column layout.  If None, derived from metadata.
        age_median: Mean age computed on training data, used to impute NaN
            ages.  If None, computed from metadata (use on training data only).

    Returns:
        X_cov (ndarray, shape [n, 1 + 2 + 2 + n_ancestry]):
            Dense covariate matrix: [intercept, age, age_missing, sex_M,
            sex_unknown, ancestry dummies...].
        ancestry_categories (list[str]):
            Ancestry category labels used (always includes "Missing").
        age_median (float):
            Mean used for age imputation (pass to subsequent calls).
    """
    # --- age: impute NaN with training mean + missing indicator ---
    age_raw = metadata['age'].values.astype(float)
    if age_median is None:
        age_median = float(np.nanmean(age_raw))
    age_missing = np.isnan(age_raw).astype(float)
    age = np.where(age_missing.astype(bool), age_median, age_raw)

    # --- sex: known-male + unknown indicators ---
    sex_vals = metadata['sex']
    sex_M = (sex_vals == 'M').astype(float).values
    sex_unknown = sex_vals.isna().astype(float).values

    # --- ancestry: one-hot with explicit "Missing" category ---
    ancestry_raw = metadata['ancestry'].fillna(_ANCESTRY_MISSING)
    if 'ancestry' in metadata.columns:
        ancestry_raw = ancestry_raw.str.strip().replace('', _ANCESTRY_MISSING)
    else:
        ancestry_raw = ancestry_raw.map(lambda _: _ANCESTRY_MISSING)

    if ancestry_categories is None:
        ancestry_categories = sorted(ancestry_raw.unique().tolist())
        # Ensure "Missing" is always present for consistent column layout
        if _ANCESTRY_MISSING not in ancestry_categories:
            ancestry_categories = sorted(ancestry_categories + [_ANCESTRY_MISSING])

    ancestry_dummies = np.zeros((len(metadata), len(ancestry_categories)), dtype=float)
    for i, cat in enumerate(ancestry_categories):
        ancestry_dummies[:, i] = (ancestry_raw.values == cat).astype(float)

    intercept = np.ones(len(metadata), dtype=float)
    X_cov = np.column_stack([intercept, age, age_missing, sex_M, sex_unknown, ancestry_dummies])
    return X_cov, ancestry_categories, age_median


class CovariateResidualizer:
    """
    OLS-based covariate residualization for arbitrary feature matrices.

    Fit on training data only; call transform() with the same fitted object
    to residualize both training and test features using identical coefficients,
    which prevents any test-set information from influencing the residuals.

    The regression fitted is:
        Y ~ X_cov   (one OLS solution per feature column)

    Residuals are:
        Y_residual = Y - X_cov @ B

    where B = lstsq(X_cov_train, Y_train).  Because an intercept column is
    included in X_cov, the training residuals are centred.

    Missing demographics are handled via indicator encoding (see
    encode_covariates), so no rows are dropped.

    Sparse input feature matrices are supported and converted to dense
    internally; the returned residuals are always dense ndarrays.
    """

    def __init__(self):
        self._ancestry_categories = None
        self._age_median = None
        self._B = None          # (n_cov, n_features) OLS coefficients

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, covariate_df, feature_matrix):
        """
        Fit OLS regression coefficients on training data.

        Args:
            covariate_df: DataFrame with 'age', 'sex', and optionally
                'ancestry' columns (one row per training sample).
            feature_matrix: ndarray or scipy sparse matrix of shape
                (n_train, n_features).  Each column is treated as an
                independent regression target.

        Returns:
            self
        """
        X_cov, self._ancestry_categories, self._age_median = encode_covariates(
            covariate_df
        )
        Y = self._to_dense(feature_matrix)
        self._B, _, _, _ = np.linalg.lstsq(X_cov, Y, rcond=None)
        return self

    def transform(self, covariate_df, feature_matrix):
        """
        Residualize feature_matrix using coefficients fitted during fit().

        Args:
            covariate_df: DataFrame with demographic columns (one row per
                sample; can differ from training data).
            feature_matrix: ndarray or sparse matrix of shape
                (n_samples, n_features).  Must have the same number of
                feature columns as the matrix passed to fit().

        Returns:
            ndarray of shape (n_samples, n_features): residuals.
        """
        if self._B is None:
            raise RuntimeError("Call fit() before transform().")
        X_cov, _, _ = encode_covariates(
            covariate_df, self._ancestry_categories, self._age_median
        )
        Y = self._to_dense(feature_matrix)
        return Y - X_cov @ self._B

    def fit_transform(self, covariate_df, feature_matrix):
        """Fit and transform training data in one step."""
        self.fit(covariate_df, feature_matrix)
        return self.transform(covariate_df, feature_matrix)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dense(X):
        if sp.issparse(X):
            return np.asarray(X.todense())
        return np.asarray(X)


# ---------------------------------------------------------------------------
# High-level helpers shared by all eval scripts
# ---------------------------------------------------------------------------

def filter_complete_demographics(df):
    """Keep rows that have at least age or sex available.

    With missing-indicator encoding, no samples need to be dropped due to
    incomplete demographics.  Only rows where age AND sex are both absent
    (truly no usable demographic info) are excluded.
    """
    return df[df['age'].notna() | df['sex'].notna()]


def covariate_adjusted_predict(X_train, train_meta, y_train, X_test, test_meta):
    """
    Residualize features against demographics, then classify with L1 logistic regression.

    Fits OLS residualization and StandardScaler on training data only; applies
    both to test data using the same coefficients.  Missing demographic values
    are handled via indicator encoding — no samples are dropped.

    Args:
        X_train: ndarray or sparse matrix of shape (n_train, n_features).
        train_meta: DataFrame with 'age', 'sex', 'ancestry' columns (n_train rows).
        y_train: array-like of binary labels (n_train,).
        X_test: ndarray or sparse matrix of shape (n_test, n_features).
        test_meta: DataFrame with 'age', 'sex', 'ancestry' columns (n_test rows).

    Returns:
        ndarray of shape (n_test,): predicted positive-class probabilities.
    """
    residualizer = CovariateResidualizer()
    X_train_res = residualizer.fit_transform(train_meta, X_train)
    X_test_res = residualizer.transform(test_meta, X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test_res)

    # saga + elasticnet + l1_ratio=1.0 is pure L1 in sklearn 1.8+.
    # liblinear + penalty='l1' triggers a deprecation/internal-check failure in 1.8.
    clf = LogisticRegression(C=1.0, solver='saga', penalty='elasticnet',
                             l1_ratio=1.0, max_iter=1000)
    clf.fit(X_train_sc, y_train)
    return clf.predict_proba(X_test_sc)[:, 1]
