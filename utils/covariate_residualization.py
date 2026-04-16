"""
Covariate residualization utility.

Fits an OLS regression predicting each feature from demographic covariates
(age, sex, ancestry) on the training fold, then returns the residuals as
de-confounded features.  Covariate encoding matches the one in
DemographicFeaturesEvaluator.featurize() exactly.

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


def encode_covariates(metadata, ancestry_categories=None):
    """
    Encode demographic columns into a numeric matrix.

    Mirrors DemographicFeaturesEvaluator.featurize():
      - age:      raw float
      - sex:      binary (M=1, F=0)
      - ancestry: one-hot over sorted unique categories seen in training

    An intercept column (all ones) is prepended so the linear regression
    absorbs the per-feature mean, ensuring residuals are centered.

    Args:
        metadata: DataFrame with 'age', 'sex', 'ancestry' columns.
            Rows must already be filtered to those with complete demographics.
        ancestry_categories: Sorted list of ancestry strings.  Pass the list
            returned by a previous call on training data so that test data
            uses the same encoding.  If None, derived from *metadata*.

    Returns:
        X_cov (ndarray, shape [n, 1 + 2 + n_ancestry]):
            Dense covariate matrix with intercept prepended.
        ancestry_categories (list[str]):
            The ancestry categories used (pass to subsequent calls for
            consistent encoding).
    """
    if ancestry_categories is None:
        ancestry_categories = sorted(metadata['ancestry'].dropna().unique().tolist())

    age = metadata['age'].values.astype(float)
    sex = (metadata['sex'] == 'M').astype(float).values

    n_ancestry = len(ancestry_categories)
    ancestry_dummies = np.zeros((len(metadata), n_ancestry), dtype=float)
    for i, cat in enumerate(ancestry_categories):
        ancestry_dummies[:, i] = (metadata['ancestry'].values == cat).astype(float)

    # Intercept first, then age, sex, ancestry dummies
    intercept = np.ones(len(metadata), dtype=float)
    X_cov = np.column_stack([intercept, age, sex, ancestry_dummies])
    return X_cov, ancestry_categories


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

    Sparse input feature matrices are supported and converted to dense
    internally; the returned residuals are always dense ndarrays.
    """

    def __init__(self):
        self._ancestry_categories = None
        self._B = None                  # (n_cov, n_features) OLS coefficients

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, covariate_df, feature_matrix):
        """
        Fit OLS regression coefficients on training data.

        Args:
            covariate_df: DataFrame with 'age', 'sex', 'ancestry' columns
                (one row per training sample).
            feature_matrix: ndarray or scipy sparse matrix of shape
                (n_train, n_features).  Each column is treated as an
                independent regression target.

        Returns:
            self
        """
        X_cov, self._ancestry_categories = encode_covariates(covariate_df)
        Y = self._to_dense(feature_matrix)
        # B: (n_cov, n_features)  via least-squares
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
        X_cov, _ = encode_covariates(covariate_df, self._ancestry_categories)
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
    """Drop rows with missing or blank age, sex, or ancestry."""
    df = df.dropna(subset=['age', 'sex', 'ancestry'])
    return df[df['ancestry'].str.strip() != '']


def covariate_adjusted_predict(X_train, train_meta, y_train, X_test, test_meta):
    """
    Residualize features against demographics, then classify with L1 logistic regression.

    Fits OLS residualization and StandardScaler on training data only; applies
    both to test data using the same coefficients.

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

    clf = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000)
    clf.fit(X_train_sc, y_train)
    return clf.predict_proba(X_test_sc)[:, 1]
