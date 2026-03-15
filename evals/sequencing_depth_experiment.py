"""
Sequencing Depth Experiment.

Evaluates TCR disease classification model performance at varying sequencing
depths by subsampling reads from repertoire files before models process them.

Two sampling modes are supported:
  - Fraction mode (--depths): subsample by fraction of each repertoire's reads.
  - Count mode (--n_seqs): subsample by absolute number of sequences. Only
    repertoires with >= --min_sequences reads are included, ensuring all depth
    levels evaluate the same set of repertoires.

Usage (fraction mode):
    python evals/sequencing_depth_experiment.py \\
        --model emerson_2017 \\
        --target_disease CMV \\
        --metadata_path data/metadata.tsv \\
        --repertoire_data_dir /path/to/data \\
        --depths 0.01 0.05 0.1 0.2 0.5 1.0 \\
        --n_repeats 3 \\
        --random_seed 42 \\
        --output_csv results/depth_experiment.csv

Usage (count mode):
    python evals/sequencing_depth_experiment.py \\
        --model emerson_2017 \\
        --target_disease CMV \\
        --metadata_path data/metadata.tsv \\
        --repertoire_data_dir /path/to/data \\
        --n_seqs 1000 5000 10000 20000 50000 100000 \\
        --min_sequences 100000 \\
        --n_repeats 3 \\
        --random_seed 42 \\
        --output_csv results/depth_experiment_counts.csv
"""

import argparse
import gzip
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


from evals.emerson_2017_disease_classification import Emerson2017Evaluator
from evals.ostmeyer_2019_disease_classification import Ostmeyer2019Evaluator
from evals.giana_2020_disease_classification import GIANA2020Evaluator


def create_evaluator(model_name, subsample_fraction=1.0, subsample_n=None,
                     subsample_seed=7, giana_dir=None):
    """
    Create an evaluator instance for the specified model with subsampling params.

    Args:
        model_name: One of 'emerson_2017', 'ostmeyer_2019', 'giana_2020'
        subsample_fraction: Fraction of reads to keep (0.0 to 1.0); ignored if subsample_n set
        subsample_n: Absolute number of reads to keep (overrides subsample_fraction if set)
        subsample_seed: Random seed for reproducible subsampling
        giana_dir: Path to GIANA directory (only needed for giana_2020)

    Returns:
        Evaluator instance
    """
    if model_name == 'emerson_2017':
        return Emerson2017Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
            subsample_n=subsample_n
        )
    elif model_name == 'ostmeyer_2019':
        return Ostmeyer2019Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
            subsample_n=subsample_n
        )
    elif model_name == 'giana_2020':
        return GIANA2020Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
            subsample_n=subsample_n,
            giana_dir=giana_dir
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: emerson_2017, ostmeyer_2019, giana_2020")


def count_sequences(file_path):
    """
    Count the number of sequences (data rows) in a TSV or TSV.GZ repertoire file.

    Args:
        file_path: Path to the repertoire file (.tsv or .tsv.gz)

    Returns:
        Number of data rows (excluding header), or 0 on error
    """
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                return sum(1 for _ in f) - 1  # subtract header
        else:
            with open(file_path, 'r') as f:
                return sum(1 for _ in f) - 1
    except Exception as e:
        print(f"Warning: could not count sequences in {file_path}: {e}")
        return 0


def get_allowed_participants(metadata_path, data_dir, min_sequences,
                             participant_col='participant_label',
                             file_prefix='part_table_', file_suffix='.tsv.gz'):
    """
    Return the set of participant labels whose repertoire files contain at least
    min_sequences reads.  Files that do not exist on disk are excluded silently.

    Args:
        metadata_path: Path to metadata TSV
        data_dir: Root directory containing repertoire files
        min_sequences: Minimum sequence count threshold
        participant_col: Column with participant labels
        file_prefix: File name prefix
        file_suffix: File name suffix

    Returns:
        Set of participant label strings that pass the threshold
    """
    metadata = pd.read_csv(metadata_path, sep='\t')
    participants = metadata[participant_col].unique()

    allowed = set()
    print(f"\nFiltering to repertoires with >= {min_sequences:,} sequences "
          f"({len(participants)} total)...")
    for participant in tqdm(participants, desc="Counting sequences"):
        file_path = os.path.join(data_dir, f"{file_prefix}{participant}{file_suffix}")
        if not os.path.exists(file_path):
            continue
        n = count_sequences(file_path)
        if n >= min_sequences:
            allowed.add(participant)

    print(f"  -> {len(allowed)} of {len(participants)} participants "
          f"have >= {min_sequences:,} sequences and will be used for all depth levels.")
    return allowed


def _run_one_depth(model_name, target_disease, metadata_path, repertoire_data_dir,
                   depth_label, subsample_fraction, subsample_n,
                   subsample_seed, random_seed, giana_dir, allowed_participants):
    """
    Run full cross-validation for a single depth/seed combination.

    Returns:
        cv_results dict from evaluator.run_cross_validation
    """
    evaluator = create_evaluator(
        model_name,
        subsample_fraction=subsample_fraction,
        subsample_n=subsample_n,
        subsample_seed=subsample_seed,
        giana_dir=giana_dir
    )
    return evaluator.run_cross_validation(
        metadata_path=metadata_path,
        target_disease=target_disease,
        data_dir=repertoire_data_dir,
        random_state=random_seed,
        tune_parameters=True,
        allowed_participants=allowed_participants
    )


def run_depth_experiment(model_name, target_disease, metadata_path, repertoire_data_dir,
                         depths=None, n_seqs=None, min_sequences=100000,
                         n_repeats=3, random_seed=7,
                         output_csv=None, giana_dir=None):
    """
    Run the sequencing depth experiment.

    For each depth x repeat combination, runs full 3-fold cross-validation
    and records AUROC/AUPR metrics.

    Either `depths` (fraction mode) or `n_seqs` (count mode) must be provided,
    but not both.

    Args:
        model_name: Model identifier string
        target_disease: Disease to classify against healthy
        metadata_path: Path to metadata TSV
        repertoire_data_dir: Directory with repertoire files
        depths: List of float fractions (fraction mode)
        n_seqs: List of int absolute sequence counts (count mode)
        min_sequences: Minimum repertoire size to include in count mode (default: 100000)
        n_repeats: Number of random subsampling repeats per depth (default: 3)
        random_seed: Base random seed (default: 7)
        output_csv: Path to save detailed CSV results (optional)
        giana_dir: Path to GIANA directory (for giana_2020 model)

    Returns:
        DataFrame with all experiment results
    """
    if (depths is None) == (n_seqs is None):
        raise ValueError("Provide exactly one of 'depths' (fraction mode) or "
                         "'n_seqs' (count mode).")

    use_count_mode = n_seqs is not None

    # Pre-compute allowed participants once for count mode
    allowed_participants = None
    if use_count_mode:
        allowed_participants = get_allowed_participants(
            metadata_path, repertoire_data_dir, min_sequences
        )
        if not allowed_participants:
            raise RuntimeError(
                f"No repertoires found with >= {min_sequences:,} sequences. "
                "Lower --min_sequences or check your data directory."
            )

    depth_levels = n_seqs if use_count_mode else depths
    all_results = []

    for depth in depth_levels:
        # At full depth (fraction=1.0 or n >= min_sequences), no randomness — run 1 repeat
        if use_count_mode:
            is_full_depth = depth >= min_sequences
        else:
            is_full_depth = depth >= 1.0
        effective_repeats = 1 if is_full_depth else n_repeats

        for repeat_idx in range(effective_repeats):
            subsample_seed = random_seed + repeat_idx

            if use_count_mode:
                depth_str = f"N={depth:,}"
            else:
                depth_str = f"{depth:.0%}"

            print(f"\n{'#'*70}")
            print(f"# DEPTH={depth_str}, REPEAT={repeat_idx+1}/{effective_repeats}, "
                  f"SEED={subsample_seed}")
            print(f"{'#'*70}")

            start_time = time.time()

            scores_df = _run_one_depth(
                model_name=model_name,
                target_disease=target_disease,
                metadata_path=metadata_path,
                repertoire_data_dir=repertoire_data_dir,
                depth_label=depth,
                subsample_fraction=depth if not use_count_mode else 1.0,
                subsample_n=depth if use_count_mode else None,
                subsample_seed=subsample_seed,
                random_seed=random_seed,
                giana_dir=giana_dir,
                allowed_participants=allowed_participants
            )

            elapsed = time.time() - start_time

            fold_col = 'malid_cross_validation_fold_id_when_in_test_set'
            fold_groups = scores_df.groupby(fold_col)
            n_folds_run = scores_df[fold_col].nunique()

            # Extract per-fold results
            for fold_val, fold_df in fold_groups:
                fold_auroc = roc_auc_score(fold_df['disease_label'], fold_df['model_score'])
                fold_aupr = average_precision_score(fold_df['disease_label'], fold_df['model_score'])
                all_results.append({
                    'model': model_name,
                    'target_disease': target_disease,
                    'sampling_mode': 'count' if use_count_mode else 'fraction',
                    'depth': depth,
                    'repeat': repeat_idx,
                    'subsample_seed': subsample_seed,
                    'fold': fold_val,
                    'test_auroc': fold_auroc,
                    'test_aupr': fold_aupr,
                    'n_test': len(fold_df),
                    'elapsed_seconds': elapsed / n_folds_run,
                })

            # Also store overall (all-folds-combined) result
            overall_auroc = roc_auc_score(scores_df['disease_label'], scores_df['model_score'])
            overall_aupr = average_precision_score(scores_df['disease_label'], scores_df['model_score'])
            all_results.append({
                'model': model_name,
                'target_disease': target_disease,
                'sampling_mode': 'count' if use_count_mode else 'fraction',
                'depth': depth,
                'repeat': repeat_idx,
                'subsample_seed': subsample_seed,
                'fold': 'overall',
                'test_auroc': overall_auroc,
                'test_aupr': overall_aupr,
                'n_test': len(scores_df),
                'elapsed_seconds': elapsed,
            })

    results_df = pd.DataFrame(all_results)

    # Save to CSV if requested
    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to: {output_csv}")

    # Print summary table
    print_summary(results_df, use_count_mode=use_count_mode)

    return results_df


def print_summary(results_df, use_count_mode=False):
    """
    Print a summary table of mean +/- std AUROC/AUPR per depth.

    Args:
        results_df: DataFrame with experiment results
        use_count_mode: Whether depths are absolute counts (True) or fractions (False)
    """
    # Filter to 'overall' fold entries only for the summary
    overall = results_df[results_df['fold'] == 'overall'].copy()

    print(f"\n{'='*70}")
    print("SEQUENCING DEPTH EXPERIMENT SUMMARY")
    if use_count_mode:
        print("Mode: absolute sequence count")
    else:
        print("Mode: fraction of repertoire")
    print(f"{'='*70}")
    print(f"{'Depth':>10s} | {'N_repeats':>9s} | {'AUROC':>16s} | {'AUPR':>16s}")
    print(f"{'-'*10}-+-{'-'*9}-+-{'-'*16}-+-{'-'*16}")

    for depth in sorted(overall['depth'].unique()):
        subset = overall[overall['depth'] == depth]
        n = len(subset)

        auroc_mean = subset['test_auroc'].mean()
        auroc_std = subset['test_auroc'].std() if n > 1 else 0.0

        aupr_mean = subset['test_aupr'].mean()
        aupr_std = subset['test_aupr'].std() if n > 1 else 0.0

        if use_count_mode:
            depth_label = f"{int(depth):>10,}"
        else:
            depth_label = f"{depth:>9.0%}"

        print(f"{depth_label} | {n:>9d} | "
              f"{auroc_mean:.4f} +/- {auroc_std:.4f} | "
              f"{aupr_mean:.4f} +/- {aupr_std:.4f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequencing Depth Experiment: evaluate model performance "
                    "at varying sequencing depths"
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['emerson_2017', 'ostmeyer_2019', 'giana_2020'],
                        help='Model to evaluate')
    parser.add_argument('--target_disease', type=str, required=True,
                        help='Target disease to classify (e.g., CMV, Lupus, T1D)')
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to metadata.tsv file')
    parser.add_argument('--repertoire_data_dir', type=str, required=True,
                        help='Root directory containing repertoire data files')

    # Mutually exclusive sampling mode
    depth_group = parser.add_mutually_exclusive_group(required=True)
    depth_group.add_argument('--depths', type=float, nargs='+',
                             help='Fraction-mode: sequencing depth fractions to test, '
                                  'e.g. 0.01 0.05 0.1 0.2 0.5 1.0')
    depth_group.add_argument('--n_seqs', type=int, nargs='+',
                             help='Count-mode: absolute number of sequences to subsample to, '
                                  'e.g. 1000 5000 10000 20000 50000 100000')

    parser.add_argument('--min_sequences', type=int, default=100000,
                        help='Count-mode only: minimum repertoire size to include '
                             '(default: 100000). Repertoires below this threshold are '
                             'excluded from all depth levels.')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of random subsampling repeats per depth '
                             '(default: 3). Ignored for full-depth level.')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Base random seed (default: 7)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save detailed results CSV')
    parser.add_argument('--giana_dir', type=str, default=None,
                        help='Path to GIANA installation directory '
                             '(only needed for giana_2020 model)')

    args = parser.parse_args()

    # Validate fraction depths
    if args.depths is not None:
        for d in args.depths:
            if not (0.0 < d <= 1.0):
                parser.error(f"Depth {d} must be in range (0.0, 1.0]")

    # Validate count depths
    if args.n_seqs is not None:
        for n in args.n_seqs:
            if n <= 0:
                parser.error(f"--n_seqs value {n} must be a positive integer")
        if max(args.n_seqs) > args.min_sequences:
            parser.error(
                f"All --n_seqs values must be <= --min_sequences ({args.min_sequences:,}). "
                f"Got max n_seqs={max(args.n_seqs):,}."
            )

    run_depth_experiment(
        model_name=args.model,
        target_disease=args.target_disease,
        metadata_path=args.metadata_path,
        repertoire_data_dir=args.repertoire_data_dir,
        depths=sorted(args.depths) if args.depths else None,
        n_seqs=sorted(args.n_seqs) if args.n_seqs else None,
        min_sequences=args.min_sequences,
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
        output_csv=args.output_csv,
        giana_dir=args.giana_dir
    )
