"""
Sequencing Depth Experiment for Mal-ID-Lite.

Mal-ID-Lite doesn't fit sequencing_depth_experiment.py's per-evaluator
`indices_map` API (it runs as a separate multi-stage subprocess pipeline via
evals.mal_id_lite_disease_classification, not an in-process evaluator class),
so depth scaling is handled at the file level instead: for each (depth,
repeat), every allowed specimen's repertoire file is filtered down to the
same pre-computed row indices the other methods use (same nesting property,
same reproducibility), written to a scoped temp directory, and fed through
the normal Mal-ID-Lite pipeline unmodified.

Reuses evals.sequencing_depth_experiment's index-loading helpers and
utils.repertoire_io.load_raw_repertoire's `subsample_indices` row-selection
(`df.iloc[indices]` on the raw, unfiltered file) so the exact same rows are
selected as for every other method at a given (depth, repeat).

Like evals.mal_id_lite_disease_classification (whose run_pipeline() this
module calls into), this requires Mal-ID-Lite to be present at
models/Mal-ID-Lite -- it is downloaded separately, not vendored as a git
submodule; see the README's Setup section.

Usage:
    python -m evals.mal_id_lite_depth_experiment \\
        --target_disease Lupus \\
        --metadata_path data/staged/malid/metadata.tsv \\
        --repertoire_data_dir data/staged/malid/repertoires \\
        --depth_indices data/Mal-ID/scaling_exp_depth_indices_max75k.json.gz \\
        --cache_root results/malid_lite_depth_cache \\
        --output_json results/malid_lite_depth_experiment.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

from evals.sequencing_depth_experiment import (
    load_depth_indices, get_allowed_specimens, build_indices_map,
)
from evals.mal_id_lite_disease_classification import (
    load_metadata, consolidate_participant_files, build_cache_if_missing,
    run_training, find_predictions_csv, convert_to_standard_scores,
    HEALTHY_LABEL,
)
from utils.repertoire_io import load_raw_repertoire


def build_depth_filtered_repertoires(
    metadata, repertoire_data_dir, indices_map, out_dir,
    participant_col="participant_label", file_prefix="part_table_", file_suffix=".tsv.gz",
):
    """Write one filtered per-specimen file per row of `metadata` under
    out_dir, keeping only the rows at indices_map[rep_id] (rep_id =
    file_prefix + participant_label + "_" + specimen_label, no suffix --
    matching sequencing_depth_experiment.py's own rep_id convention).
    Specimens absent from indices_map are skipped (not in this depth-indices
    file's coverage).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_gz = file_suffix.endswith(".gz")
    written, skipped = 0, 0

    for _, row in metadata.iterrows():
        participant, specimen = row[participant_col], row["specimen_label"]
        rep_id = f"{file_prefix}{participant}_{specimen}"
        indices = indices_map.get(rep_id)
        if indices is None:
            skipped += 1
            continue
        src = Path(repertoire_data_dir) / f"{rep_id}{file_suffix}"
        df = load_raw_repertoire(str(src), subsample_indices=indices)
        if df.empty:
            skipped += 1
            continue
        out_path = out_dir / f"{rep_id}{file_suffix}"
        df.to_csv(out_path, sep="\t", index=False, compression="gzip" if is_gz else None)
        written += 1

    print(f"  Depth-filtered repertoires: {written} written, {skipped} skipped "
          f"(missing from indices file or empty after filtering).")
    return written


def run_one_depth_repeat(
    depth, repeat_idx, metadata, allowed_specimens, repertoire_data_dir,
    repertoires_index, target_disease, dataset_name, cache_root,
    participant_col, disease_col, fold_col, file_prefix, file_suffix,
    healthy_label, gene_locus, models, model2_abstention_strategy, n_jobs, use_gpu,
):
    tag = f"depth{depth}_rep{repeat_idx}"
    scoped = Path(cache_root) / tag
    filtered_dir = scoped / "filtered_repertoires"
    cache_dir = scoped / "cache"
    output_dir = scoped / "output"

    subset_metadata = metadata[metadata["specimen_label"].isin(allowed_specimens)].copy()
    indices_map = build_indices_map(repertoires_index, repeat_idx, depth)

    print(f"  [{tag}] Building depth-filtered repertoires ...")
    build_depth_filtered_repertoires(
        subset_metadata, repertoire_data_dir, indices_map, filtered_dir,
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )

    target_subset = subset_metadata[
        subset_metadata[disease_col].isin([target_disease, healthy_label])
    ].copy()

    print(f"  [{tag}] Consolidating to per-participant files ...")
    has_nt_cdr3 = consolidate_participant_files(
        target_subset, filtered_dir, cache_dir / "consolidated_data",
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )

    staged_metadata_path = cache_dir / "staged_metadata_for_malid_lite.tsv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    renamed = target_subset.rename(columns={disease_col: "disease"}) if disease_col != "disease" else target_subset
    renamed.to_csv(staged_metadata_path, sep="\t", index=False)

    print(f"  [{tag}] Building Mal-ID-Lite cache (skipped if already built) ...")
    build_cache_if_missing(
        cache_dir / "consolidated_data", staged_metadata_path, cache_dir, gene_locus,
        n_jobs, not has_nt_cdr3, force_reprocess=False,
    )

    print(f"  [{tag}] Training ...")
    run_training(
        cache_dir, f"{dataset_name}_{tag}", gene_locus, target_disease, healthy_label,
        models, model2_abstention_strategy, output_dir, n_jobs, use_gpu,
    )

    predictions_csv = find_predictions_csv(output_dir)
    scores_df = convert_to_standard_scores(predictions_csv, target_disease)
    return scores_df


def run_depth_experiment(
    target_disease, metadata_path, repertoire_data_dir, depth_indices_path,
    dataset_name, cache_root, output_json=None,
    participant_col="participant_label", disease_col="disease",
    fold_col="malid_cross_validation_fold_id_when_in_test_set",
    file_prefix="part_table_", file_suffix=".tsv.gz", healthy_label=HEALTHY_LABEL,
    gene_locus="TCR", models=(1, 2, 3), model2_abstention_strategy="fill_models13_mean",
    n_jobs=4, use_gpu=True,
):
    print(f"Loading depth indices from: {depth_indices_path}")
    index_data = load_depth_indices(depth_indices_path)
    depths = index_data["depths"]
    n_repeats = index_data["n_reps"]
    repertoires_index = index_data["repertoires"]

    print(f"  Depths: {depths}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Repertoires: {len(repertoires_index)}")

    metadata = load_metadata(metadata_path)
    allowed_specimens = get_allowed_specimens(
        metadata_path, repertoires_index, participant_col=participant_col, file_prefix=file_prefix,
    )
    print(f"  Specimens matched in metadata: {len(allowed_specimens)}")

    all_results = []
    for depth in depths:
        for repeat_idx in range(n_repeats):
            print(f"\n{'#'*70}")
            print(f"# DEPTH=N={depth:,}, REPEAT={repeat_idx+1}/{n_repeats}")
            print(f"{'#'*70}")
            start_time = time.time()

            scores_df = run_one_depth_repeat(
                depth, repeat_idx, metadata, allowed_specimens, repertoire_data_dir,
                repertoires_index, target_disease, dataset_name, cache_root,
                participant_col, disease_col, fold_col, file_prefix, file_suffix,
                healthy_label, gene_locus, list(models), model2_abstention_strategy, n_jobs, use_gpu,
            )
            elapsed = time.time() - start_time

            y = scores_df["disease_label"].values
            p = scores_df["model_score"].values
            preds = (p >= 0.5).astype(int)
            all_results.append({
                "depth": depth,
                "repeat": repeat_idx,
                "auroc": roc_auc_score(y, p),
                "aupr": average_precision_score(y, p),
                "balanced_acc": balanced_accuracy_score(y, preds),
                "f1": f1_score(y, preds),
                "n_samples": len(scores_df),
                "elapsed_seconds": round(elapsed, 2),
            })

    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "model": "malid_lite",
            "target_disease": target_disease,
            "depth_indices_path": depth_indices_path,
            "n_repeats": n_repeats,
            "depths": depths,
            "malid_lite_params": {
                "models": list(models),
                "model2_abstention_strategy": model2_abstention_strategy,
                "gene_locus": gene_locus,
            },
            "results": all_results,
        }
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mal-ID-Lite Sequencing Depth Experiment")
    parser.add_argument("--target_disease", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--repertoire_data_dir", type=str, required=True)
    parser.add_argument("--depth_indices", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True,
                        help="Root directory for per-(depth,repeat) filtered data/cache/output "
                             "(scoped subdirectories are created underneath).")
    parser.add_argument("--participant_col", type=str, default="participant_label")
    parser.add_argument("--disease_col", type=str, default="disease")
    parser.add_argument("--fold_col", type=str,
                        default="malid_cross_validation_fold_id_when_in_test_set")
    parser.add_argument("--file_prefix", type=str, default="part_table_")
    parser.add_argument("--file_suffix", type=str, default=".tsv.gz")
    parser.add_argument("--healthy_label", type=str, default=HEALTHY_LABEL)
    parser.add_argument("--gene_locus", type=str, default="TCR", choices=["TCR"])
    parser.add_argument("--models", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--model2_abstention_strategy", type=str, default="fill_models13_mean")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    run_depth_experiment(
        target_disease=args.target_disease,
        metadata_path=args.metadata_path,
        repertoire_data_dir=args.repertoire_data_dir,
        depth_indices_path=args.depth_indices,
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        output_json=args.output_json,
        participant_col=args.participant_col,
        disease_col=args.disease_col,
        fold_col=args.fold_col,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
        healthy_label=args.healthy_label,
        gene_locus=args.gene_locus,
        models=args.models,
        model2_abstention_strategy=args.model2_abstention_strategy,
        n_jobs=args.n_jobs,
        use_gpu=not args.no_gpu,
    )
