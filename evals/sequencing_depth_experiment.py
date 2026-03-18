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
        --n_repeats 5 \\
        --random_seed 42 \\
        --output_json results/depth_experiment.json

Usage (count mode):
    python evals/sequencing_depth_experiment.py \\
        --model emerson_2017 \\
        --target_disease CMV \\
        --metadata_path data/metadata.tsv \\
        --repertoire_data_dir /path/to/data \\
        --n_seqs 1000 5000 10000 20000 50000 100000 \\
        --min_sequences 100000 \\
        --n_repeats 5 \\
        --random_seed 42 \\
        --output_json results/depth_experiment_counts.json
"""

import argparse
import gzip
import json
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
    Return the set of specimen labels whose repertoire files contain at least
    min_sequences reads.  Files that do not exist on disk are excluded silently.

    Args:
        metadata_path: Path to metadata TSV
        data_dir: Root directory containing repertoire files
        min_sequences: Minimum sequence count threshold
        participant_col: Column with participant labels
        file_prefix: File name prefix
        file_suffix: File name suffix

    Returns:
        Set of specimen label strings that pass the threshold
    """
    metadata = pd.read_csv(metadata_path, sep='\t')
    specimens = metadata[[participant_col, 'specimen_label']].drop_duplicates()

    allowed = set()
    print(f"\nFiltering to repertoires with >= {min_sequences:,} sequences "
          f"({len(specimens)} total)...")
    for _, row in tqdm(specimens.iterrows(), total=len(specimens), desc="Counting sequences"):
        file_path = os.path.join(
            data_dir,
            f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}"
        )
        if not os.path.exists(file_path):
            continue
        n = count_sequences(file_path)
        if n >= min_sequences:
            allowed.add(row['specimen_label'])

    print(f"  -> {len(allowed)} of {len(specimens)} specimens "
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
                         n_repeats=5, random_seed=7,
                         output_json=None, giana_dir=None):
    """
    Run the sequencing depth experiment.

    For each depth x repeat combination, runs full 3-fold cross-validation
    and records overall AUROC/AUPR metrics.

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
        n_repeats: Number of random subsampling repeats per depth (default: 5)
        random_seed: Base random seed (default: 7)
        output_json: Path to save results JSON (optional)
        giana_dir: Path to GIANA directory (for giana_2020 model)

    Returns:
        List of result dicts (one per depth x repeat)
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

            overall_auroc = roc_auc_score(scores_df['disease_label'], scores_df['model_score'])
            overall_aupr = average_precision_score(scores_df['disease_label'], scores_df['model_score'])
            all_results.append({
                'depth': depth,
                'repeat': repeat_idx,
                'subsample_seed': subsample_seed,
                'auroc': overall_auroc,
                'aupr': overall_aupr,
                'n_samples': len(scores_df),
                'elapsed_seconds': round(elapsed, 2),
            })

    # Save JSON if requested
    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = {
            'model': model_name,
            'target_disease': target_disease,
            'sampling_mode': 'count' if use_count_mode else 'fraction',
            'n_repeats': n_repeats,
            'random_seed': random_seed,
            'results': all_results,
        }
        if use_count_mode:
            output_data['min_sequences'] = min_sequences
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    # Print summary table
    print_summary(all_results, use_count_mode=use_count_mode)

    return all_results


def print_summary(results, use_count_mode=False):
    """
    Print a summary table of mean +/- stderr AUROC/AUPR per depth.

    Args:
        results: List of result dicts with 'depth', 'auroc', 'aupr' keys
        use_count_mode: Whether depths are absolute counts (True) or fractions (False)
    """
    results_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("SEQUENCING DEPTH EXPERIMENT SUMMARY")
    if use_count_mode:
        print("Mode: absolute sequence count")
    else:
        print("Mode: fraction of repertoire")
    print(f"{'='*70}")
    print(f"{'Depth':>10s} | {'N':>3s} | {'AUROC':>18s} | {'AUPR':>18s}")
    print(f"{'-'*10}-+-{'-'*3}-+-{'-'*18}-+-{'-'*18}")

    for depth in sorted(results_df['depth'].unique()):
        subset = results_df[results_df['depth'] == depth]
        n = len(subset)

        auroc_mean = subset['auroc'].mean()
        auroc_se = subset['auroc'].std() / np.sqrt(n) if n > 1 else 0.0

        aupr_mean = subset['aupr'].mean()
        aupr_se = subset['aupr'].std() / np.sqrt(n) if n > 1 else 0.0

        if use_count_mode:
            depth_label = f"{int(depth):>10,}"
        else:
            depth_label = f"{depth:>9.0%}"

        print(f"{depth_label} | {n:>3d} | "
              f"{auroc_mean:.4f} +/- {auroc_se:.4f} | "
              f"{aupr_mean:.4f} +/- {aupr_se:.4f}")

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
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='Number of random subsampling repeats per depth '
                             '(default: 5). Ignored for full-depth level.')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Base random seed (default: 7)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save results JSON')
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
        output_json=args.output_json,
        giana_dir=args.giana_dir
    )
