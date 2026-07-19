"""Helpers for pooled outer-fold evaluation."""

import os


def outer_test_folds(n_folds):
    """Return all outer folds or the single fold selected by the launcher."""
    selected = os.environ.get('AIRR_BENCH_TEST_FOLD')
    if selected is None:
        return range(n_folds)

    selected = int(selected)
    if selected < 0 or selected >= n_folds:
        raise ValueError(
            f"AIRR_BENCH_TEST_FOLD must be between 0 and {n_folds - 1}; "
            f"got {selected}")
    return [selected]


def split_metadata(metadata, fold_col, test_fold):
    """Hold out one test fold and pool every other fold for development."""
    test = metadata[metadata[fold_col] == test_fold]
    train_val_pool = metadata[metadata[fold_col] != test_fold]
    return train_val_pool, test
