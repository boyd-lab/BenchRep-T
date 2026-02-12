"""
Sequencing Depth Experiment.

Evaluates TCR disease classification model performance at varying sequencing
depths by subsampling reads from repertoire files before models process them.

Usage:
    python evals/sequencing_depth_experiment.py \
        --model emerson_2017 \
        --target_disease CMV \
        --metadata_path data/metadata.tsv \
        --repertoire_data_dir /path/to/data \
        --depths 0.01 0.05 0.1 0.2 0.5 1.0 \
        --n_repeats 3 \
        --random_seed 42 \
        --output_csv results/depth_experiment.csv
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd


from evals.emerson_2017_disease_classification import Emerson2017Evaluator
from evals.ostmeyer_2019_disease_classification import Ostmeyer2019Evaluator
from evals.giana_2020_disease_classification import GIANA2020Evaluator


def create_evaluator(model_name, subsample_fraction, subsample_seed, giana_dir=None):
    """
    Create an evaluator instance for the specified model with subsampling params.

    Args:
        model_name: One of 'emerson_2017', 'ostmeyer_2019', 'giana_2020'
        subsample_fraction: Fraction of reads to keep (0.0 to 1.0)
        subsample_seed: Random seed for reproducible subsampling
        giana_dir: Path to GIANA directory (only needed for giana_2020)

    Returns:
        Evaluator instance
    """
    if model_name == 'emerson_2017':
        return Emerson2017Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed
        )
    elif model_name == 'ostmeyer_2019':
        return Ostmeyer2019Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed
        )
    elif model_name == 'giana_2020':
        return GIANA2020Evaluator(
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
            giana_dir=giana_dir
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: emerson_2017, ostmeyer_2019, giana_2020")


def run_depth_experiment(model_name, target_disease, metadata_path, repertoire_data_dir,
                         depths=None, n_repeats=3, random_seed=7,
                         output_csv=None, giana_dir=None):
    """
    Run the sequencing depth experiment.

    For each depth x repeat combination, runs full 3-fold cross-validation
    and records AUROC/AUPR metrics.

    Args:
        model_name: Model identifier string
        target_disease: Disease to classify against healthy
        metadata_path: Path to metadata TSV
        repertoire_data_dir: Directory with repertoire files
        depths: List of float fractions (default: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        n_repeats: Number of random subsampling repeats per depth (default: 3)
        random_seed: Base random seed (default: 42)
        output_csv: Path to save detailed CSV results (optional)
        giana_dir: Path to GIANA directory (for giana_2020 model)

    Returns:
        DataFrame with all experiment results
    """
    if depths is None:
        depths = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    all_results = []

    for depth in depths:
        # At 100% depth, no randomness -- run only 1 repeat
        effective_repeats = 1 if depth >= 1.0 else n_repeats

        for repeat_idx in range(effective_repeats):
            subsample_seed = random_seed + repeat_idx

            print(f"\n{'#'*70}")
            print(f"# DEPTH={depth:.0%}, REPEAT={repeat_idx+1}/{effective_repeats}, "
                  f"SEED={subsample_seed}")
            print(f"{'#'*70}")

            start_time = time.time()

            # Create fresh evaluator for this depth/repeat
            evaluator = create_evaluator(
                model_name,
                subsample_fraction=depth,
                subsample_seed=subsample_seed,
                giana_dir=giana_dir
            )

            # Run full cross-validation
            cv_results = evaluator.run_cross_validation(
                metadata_path=metadata_path,
                target_disease=target_disease,
                data_dir=repertoire_data_dir,
                random_state=random_seed,  # Same CV splits across all depths
                tune_parameters=True
            )

            elapsed = time.time() - start_time

            # Extract per-fold results
            for fold_result in cv_results['fold_results']:
                all_results.append({
                    'model': model_name,
                    'target_disease': target_disease,
                    'depth': depth,
                    'repeat': repeat_idx,
                    'subsample_seed': subsample_seed,
                    'fold': fold_result['fold'],
                    'test_auroc': fold_result['test_auroc'],
                    'test_aupr': fold_result['test_aupr'],
                    'val_auroc': fold_result['val_auroc'],
                    'val_aupr': fold_result['val_aupr'],
                    'n_train': fold_result['n_train'],
                    'n_val': fold_result['n_val'],
                    'n_test': fold_result['n_test'],
                    'elapsed_seconds': elapsed / len(cv_results['fold_results'])
                })

            # Also store overall (all-folds-combined) result
            all_results.append({
                'model': model_name,
                'target_disease': target_disease,
                'depth': depth,
                'repeat': repeat_idx,
                'subsample_seed': subsample_seed,
                'fold': 'overall',
                'test_auroc': cv_results['overall_auroc'],
                'test_aupr': cv_results['overall_aupr'],
                'val_auroc': np.mean([r['val_auroc'] for r in cv_results['fold_results']]),
                'val_aupr': np.mean([r['val_aupr'] for r in cv_results['fold_results']]),
                'n_train': sum(r['n_train'] for r in cv_results['fold_results']),
                'n_val': sum(r['n_val'] for r in cv_results['fold_results']),
                'n_test': sum(r['n_test'] for r in cv_results['fold_results']),
                'elapsed_seconds': elapsed
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
    print_summary(results_df)

    return results_df


def print_summary(results_df):
    """
    Print a summary table of mean +/- std AUROC/AUPR per depth.

    Args:
        results_df: DataFrame with experiment results
    """
    # Filter to 'overall' fold entries only for the summary
    overall = results_df[results_df['fold'] == 'overall'].copy()

    print(f"\n{'='*70}")
    print("SEQUENCING DEPTH EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Depth':>8s} | {'N_repeats':>9s} | {'AUROC':>16s} | {'AUPR':>16s}")
    print(f"{'-'*8}-+-{'-'*9}-+-{'-'*16}-+-{'-'*16}")

    for depth in sorted(overall['depth'].unique()):
        subset = overall[overall['depth'] == depth]
        n = len(subset)

        auroc_mean = subset['test_auroc'].mean()
        auroc_std = subset['test_auroc'].std() if n > 1 else 0.0

        aupr_mean = subset['test_aupr'].mean()
        aupr_std = subset['test_aupr'].std() if n > 1 else 0.0

        print(f"{depth:>7.0%} | {n:>9d} | "
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
    parser.add_argument('--depths', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                        help='Sequencing depth fractions to test '
                             '(default: 0.01 0.05 0.1 0.2 0.5 1.0)')
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of random subsampling repeats per depth '
                             '(default: 3). Ignored for depth=1.0.')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save detailed results CSV')
    parser.add_argument('--giana_dir', type=str, default=None,
                        help='Path to GIANA installation directory '
                             '(only needed for giana_2020 model)')

    args = parser.parse_args()

    # Validate depths
    for d in args.depths:
        if not (0.0 < d <= 1.0):
            parser.error(f"Depth {d} must be in range (0.0, 1.0]")

    run_depth_experiment(
        model_name=args.model,
        target_disease=args.target_disease,
        metadata_path=args.metadata_path,
        repertoire_data_dir=args.repertoire_data_dir,
        depths=sorted(args.depths),
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
        output_csv=args.output_csv,
        giana_dir=args.giana_dir
    )