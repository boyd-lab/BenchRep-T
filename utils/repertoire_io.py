"""
Shared utility for loading repertoire TSV files.

All models should use load_raw_repertoire() for the file reading and
subsampling step to ensure consistent behavior across methods.
"""

import pandas as pd


def load_raw_repertoire(file_path, subsample_n=None, subsample_fraction=1.0,
                        subsample_seed=7, subsample_indices=None):
    """
    Load a repertoire TSV file and apply optional subsampling.

    Priority: subsample_indices > subsample_n > subsample_fraction.
    Returns an empty DataFrame on error.

    Args:
        file_path: Path to a .tsv or .tsv.gz repertoire file.
        subsample_n: Absolute number of rows to keep (default: None = keep all).
        subsample_fraction: Fraction of rows to keep, e.g. 0.5 (default: 1.0).
        subsample_seed: Random seed for reproducible subsampling (default: 7).
        subsample_indices: Pre-computed list of row indices to select (default: None).

    Returns:
        pd.DataFrame with the (possibly subsampled) repertoire rows.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if subsample_indices is not None:
            df = df.iloc[subsample_indices]
        elif subsample_n is not None:
            df = df.sample(n=min(subsample_n, len(df)), random_state=subsample_seed)
        elif subsample_fraction < 1.0:
            df = df.sample(frac=subsample_fraction, random_state=subsample_seed)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()
