"""
Sequencing Depth Experiment.

Evaluates TCR disease classification model performance at varying sequencing
depths using pre-generated subsampling indices from generate_depth_indices.py.

The indices file ensures:
  - Consistent repertoire filtering (only repertoires with enough sequences)
  - Nested subsampling (depth K uses first K indices of each repetition)
  - Reproducible random seeds via numpy SeedSequence

Usage:
    python -m evals.sequencing_depth_experiment \\
        --model emerson_2017 \\
        --target_disease CMV \\
        --metadata_path data/metadata.tsv \\
        --repertoire_data_dir /path/to/data \\
        --depth_indices data/depth_indices_seed7.json.gz \\
        --depths 1000 5000 10000 25000 50000 100000 \\
        --output_json results/depth_experiment.json
"""

import argparse
import gzip
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def create_evaluator(model_name, indices_map=None, giana_dir=None):
    """
    Create an evaluator instance for the specified model.

    Args:
        model_name: One of 'emerson_2017', 'ostmeyer_2019', 'giana_2020'
        indices_map: Dict mapping rep_id to pre-computed row indices
        giana_dir: Path to GIANA directory (only needed for giana_2020)

    Returns:
        Evaluator instance
    """
    if model_name == 'emerson_2017':
        from evals.emerson_2017_disease_classification import Emerson2017Evaluator
        return Emerson2017Evaluator(indices_map=indices_map)
    elif model_name == 'ostmeyer_2019':
        from evals.ostmeyer_2019_disease_classification import Ostmeyer2019Evaluator
        return Ostmeyer2019Evaluator(indices_map=indices_map)
    elif model_name == 'giana_2020':
        from evals.giana_2020_disease_classification import GIANA2020Evaluator
        return GIANA2020Evaluator(indices_map=indices_map, giana_dir=giana_dir)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: emerson_2017, ostmeyer_2019, giana_2020")


def load_depth_indices(path):
    """
    Load pre-generated depth indices from a JSON or JSON.GZ file.

    Args:
        path: Path to the indices file

    Returns:
        Dict: {rep_id: {str(repeat): [indices...]}}
    """
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def get_allowed_specimens(metadata_path, depth_indices,
                          participant_col='participant_label',
                          file_prefix='part_table_'):
    """
    Return the set of specimen labels whose repertoire rep_id exists in the
    depth indices file.

    Args:
        metadata_path: Path to metadata TSV
        depth_indices: Loaded depth indices dict
        participant_col: Column with participant labels
        file_prefix: File name prefix used to construct rep_id

    Returns:
        Set of specimen_label strings present in the indices file
    """
    metadata = pd.read_csv(metadata_path, sep='\t')
    specimens = metadata[[participant_col, 'specimen_label']].drop_duplicates()
    allowed = set()
    for _, row in specimens.iterrows():
        rep_id = f"{file_prefix}{row[participant_col]}_{row['specimen_label']}"
        if rep_id in depth_indices:
            allowed.add(row['specimen_label'])
    return allowed


def build_indices_map(depth_indices, repeat, depth):
    """
    Build an indices_map for a specific (depth, repeat) combination.

    Uses the nesting property: first `depth` indices of each repetition's
    pre-shuffled index array.

    Args:
        depth_indices: Full indices dict {rep_id: {str(repeat): [indices...]}}
        repeat: Repetition index (int)
        depth: Number of sequences to keep (int)

    Returns:
        Dict: {rep_id: list of row indices}
    """
    indices_map = {}
    repeat_key = str(repeat)
    for rep_id, rep_data in depth_indices.items():
        indices_map[rep_id] = rep_data[repeat_key][:depth]
    return indices_map


def run_depth_experiment(model_name, target_disease, metadata_path, repertoire_data_dir,
                         depth_indices_path, depths, random_seed=7,
                         output_json=None, giana_dir=None):
    """
    Run the sequencing depth experiment using pre-generated indices.

    Args:
        model_name: Model identifier string
        target_disease: Disease to classify against healthy
        metadata_path: Path to metadata TSV
        repertoire_data_dir: Directory with repertoire files
        depth_indices_path: Path to depth indices JSON/JSON.GZ file
        depths: List of int depth levels to evaluate
        random_seed: Random seed for train/val split (default: 7)
        output_json: Path to save results JSON (optional)
        giana_dir: Path to GIANA directory (for giana_2020 model)

    Returns:
        List of result dicts (one per depth x repeat)
    """
    # Load pre-generated indices
    print(f"Loading depth indices from: {depth_indices_path}")
    depth_indices = load_depth_indices(depth_indices_path)

    # Infer n_repeats from the indices file
    first_rep = next(iter(depth_indices.values()))
    n_repeats = len(first_rep)
    max_depth = len(next(iter(first_rep.values())))
    print(f"  Repertoires: {len(depth_indices)}, Repeats: {n_repeats}, "
          f"Max depth: {max_depth:,}")

    # Validate requested depths
    for d in depths:
        if d > max_depth:
            raise ValueError(
                f"Requested depth {d:,} exceeds max depth {max_depth:,} "
                f"in indices file."
            )

    # Determine allowed specimens from indices file
    allowed_specimens = get_allowed_specimens(metadata_path, depth_indices)
    print(f"  Specimens with indices: {len(allowed_specimens)}")

    all_results = []

    for depth in depths:
        for repeat_idx in range(n_repeats):
            print(f"\n{'#'*70}")
            print(f"# DEPTH=N={depth:,}, REPEAT={repeat_idx+1}/{n_repeats}")
            print(f"{'#'*70}")

            start_time = time.time()

            # Build indices_map for this (depth, repeat) combination
            indices_map = build_indices_map(depth_indices, repeat_idx, depth)

            # Create evaluator with the indices
            evaluator = create_evaluator(
                model_name, indices_map=indices_map, giana_dir=giana_dir
            )

            # Run cross-validation
            scores_df = evaluator.run_cross_validation(
                metadata_path=metadata_path,
                target_disease=target_disease,
                data_dir=repertoire_data_dir,
                random_state=random_seed,
                tune_parameters=True,
                allowed_participants=allowed_specimens
            )

            elapsed = time.time() - start_time

            overall_auroc = roc_auc_score(scores_df['disease_label'], scores_df['model_score'])
            overall_aupr = average_precision_score(scores_df['disease_label'], scores_df['model_score'])
            all_results.append({
                'depth': depth,
                'repeat': repeat_idx,
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
            'depth_indices_path': depth_indices_path,
            'n_repeats': n_repeats,
            'random_seed': random_seed,
            'depths': depths,
            'results': all_results,
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results):
    """
    Print a summary table of mean +/- stderr AUROC/AUPR per depth.

    Args:
        results: List of result dicts with 'depth', 'auroc', 'aupr' keys
    """
    results_df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("SEQUENCING DEPTH EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Depth':>10s} | {'N':>3s} | {'AUROC':>18s} | {'AUPR':>18s}")
    print(f"{'-'*10}-+-{'-'*3}-+-{'-'*18}-+-{'-'*18}")

    for depth in sorted(results_df['depth'].unique()):
        subset = results_df[results_df['depth'] == depth]
        n = len(subset)

        auroc_mean = subset['auroc'].mean()
        auroc_se = subset['auroc'].std() / np.sqrt(n) if n > 1 else 0.0

        aupr_mean = subset['aupr'].mean()
        aupr_se = subset['aupr'].std() / np.sqrt(n) if n > 1 else 0.0

        print(f"{int(depth):>10,} | {n:>3d} | "
              f"{auroc_mean:.4f} +/- {auroc_se:.4f} | "
              f"{aupr_mean:.4f} +/- {aupr_se:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequencing Depth Experiment: evaluate model performance "
                    "at varying sequencing depths using pre-generated indices"
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
    parser.add_argument('--depth_indices', type=str, required=True,
                        help='Path to pre-generated depth indices file '
                             '(.json or .json.gz from generate_depth_indices.py)')
    parser.add_argument('--depths', type=int, nargs='+', required=True,
                        help='Depth levels to evaluate (number of sequences), '
                             'e.g. 1000 5000 10000 25000 50000 100000')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Random seed for train/val split (default: 7)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--giana_dir', type=str, default=None,
                        help='Path to GIANA installation directory '
                             '(only needed for giana_2020 model)')

    args = parser.parse_args()

    for d in args.depths:
        if d <= 0:
            parser.error(f"Depth {d} must be a positive integer")

    run_depth_experiment(
        model_name=args.model,
        target_disease=args.target_disease,
        metadata_path=args.metadata_path,
        repertoire_data_dir=args.repertoire_data_dir,
        depth_indices_path=args.depth_indices,
        depths=sorted(args.depths),
        random_seed=args.random_seed,
        output_json=args.output_json,
        giana_dir=args.giana_dir
    )
