"""Helpers for selecting benchmark outer folds and fixed train/val/test splits."""

import os


def outer_test_folds(n_folds, fixed_split=False):
    """Return rotating folds, a selected outer fold, or the fixed Rawat split.

    ``AIRR_BENCH_TEST_FOLD`` selects one ordinary outer fold while preserving
    each evaluator's original internal train/validation split.  This differs
    from ``fixed_split``, which explicitly assigns fold 0 to train, fold 1 to
    validation, and fold 2 to test.
    """
    if fixed_split:
        return [2]

    selected = os.environ.get('AIRR_BENCH_TEST_FOLD')
    if selected is None:
        return range(n_folds)

    selected = int(selected)
    if selected < 0 or selected >= n_folds:
        raise ValueError(
            f"AIRR_BENCH_TEST_FOLD must be between 0 and {n_folds - 1}; "
            f"got {selected}")
    return [selected]


def split_metadata(metadata, fold_col, test_fold, fixed_split=False):
    """Split metadata, reserving fold 0/1/2 when fixed_split is enabled."""
    if fixed_split:
        present = set(metadata[fold_col].dropna().astype(int).unique())
        missing = {0, 1, 2} - present
        if missing:
            raise ValueError(f"Fixed split requires folds 0, 1, and 2; missing {sorted(missing)}")
        train = metadata[metadata[fold_col] == 0]
        val = metadata[metadata[fold_col] == 1]
        test = metadata[metadata[fold_col] == 2]
        return train, val, test

    test = metadata[metadata[fold_col] == test_fold]
    return None, metadata[metadata[fold_col] != test_fold], test
