"""
Sequencing Depth Experiment for Mal-ID-Lite.

Mal-ID-Lite doesn't fit sequencing_depth_experiment.py's per-evaluator
`indices_map` API (it runs as a separate multi-stage subprocess pipeline via
evals.mal_id_lite_disease_classification, not an in-process evaluator class),
so depth scaling is handled at the file level instead: for each (depth,
repeat), every allowed specimen's repertoire file is filtered down to the
same pre-computed row indices the other methods use (same nesting property,
same reproducibility), consolidated into Mal-ID-Lite's native per-participant
format, cached, embedded, and trained through the normal Mal-ID-Lite pipeline.

All diseases passed via --target_diseases are sampled, cached, and embedded
TOGETHER, once per (depth, repeat) -- not once per disease. Only the final
training step distinguishes between diseases, via Mal-ID-Lite's own
multi-binary classification mode (--classification-mode multi-binary), which
trains one independent binary ensemble per disease (each vs. the shared
--healthy_label reference class) while reusing the same cache/embeddings
across all of them. This matches how depth-scaling experiments are actually
run outside BenchRep-T, and avoids an earlier per-disease design that rebuilt
the expensive, disease-agnostic cache and ESM-2 embeddings once per disease for
the shared healthy/reference-class participants -- every disease is trained
against the same reference population, so that population's cache/embeddings
only need computing once per (depth, repeat), not once per disease. The
disease-specific participants themselves are never shared between diseases;
only the reference class is common to every pair.

clone_id is always recomputed fresh on the downsampled data, for every
(depth, repeat): the raw malid data ships with a pre-existing clone_id column
(from its own upstream cleaning pipeline, well before BenchRep-T sees it),
which subsampling would otherwise carry forward unchanged (row selection
preserves every column). A clone_id computed by clustering the *full*
repertoire is not valid for a downsampled subset of it -- which clones exist
at all, and their membership, both depend on which sequences are actually
present. build_depth_filtered_repertoires() drops this stale column
explicitly, so Mal-ID-Lite's own loader (which only computes clone_id when
the column is absent, never re-deriving it if one is already there) recomputes
it correctly from the actual downsampled rows every time.

Reuses evals.sequencing_depth_experiment's index-loading helpers and
utils.repertoire_io.load_raw_repertoire's `subsample_indices` row-selection
(`df.iloc[indices]` on the raw, unfiltered file) so the exact same rows are
selected as for every other method at a given (depth, repeat).

Like evals.mal_id_lite_disease_classification (several of whose lower-level
helper functions this module reuses directly -- consolidate_participant_files,
build_cache_if_missing, run_training, and others -- rather than its own
run_pipeline()), this requires Mal-ID-Lite to be present at
models/Mal-ID-Lite -- it is downloaded separately, not vendored as a git
submodule; see the README's Setup section.

Every (depth, repeat)'s filtered repertoires, cache, and embeddings persist on
disk indefinitely under --cache_root/depth{D}_rep{R}/ -- nothing here is ever
deleted, so re-running the same (depth, repeat) combination (e.g. resuming an
interrupted sweep) reuses what's already there instead of rebuilding it. This
reuse is keyed only by (depth, repeat), not by which diseases were requested,
so widening --target_diseases for a combination that was already cached with
a narrower set is checked explicitly and fails fast with a clear error (see
_validate_reused_depth_cache_diseases()) rather than silently training an
incomplete multi-binary ensemble.

Usage:
    python -m evals.mal_id_lite_depth_experiment \\
        --target_diseases Lupus HIV \\
        --metadata_path data/staged/malid/metadata.tsv \\
        --repertoire_data_dir data/staged/malid/repertoires \\
        --depth_indices data/Mal-ID/scaling_exp_depth_indices_max75k.json.gz \\
        --dataset_name malid \\
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
    run_training, convert_to_standard_scores,
    HEALTHY_LABEL, _rename_for_staging,
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

    Drops `clone_id` from the output if present. The raw malid data ships
    with a pre-existing clone_id column (see this module's own docstring);
    row-subsampling alone (`load_raw_repertoire`'s `subsample_indices`) would
    otherwise carry it forward unchanged onto data it no longer correctly
    describes. Dropping it here (rather than passing --force-clone-id at
    cache-build time) means Mal-ID-Lite's own loader recomputes clone_id
    because the column is genuinely absent -- the same mechanism that already
    triggers computation for cohorts that never had a clone_id column to
    begin with, no extra flag needed.
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
        if "clone_id" in df.columns:
            df = df.drop(columns=["clone_id"])
        out_path = out_dir / f"{rep_id}{file_suffix}"
        df.to_csv(out_path, sep="\t", index=False, compression="gzip" if is_gz else None)
        written += 1

    print(f"  Depth-filtered repertoires: {written} written, {skipped} skipped "
          f"(missing from indices file or empty after filtering).")
    return written


def _make_pair_dir_name(disease, reference):
    """Mirrors Mal-ID-Lite's own malid_lite.training.training_utils.make_pair_name():
    the filesystem-safe "<disease>_vs_<reference>" directory name multi-binary
    training writes each disease pair's own artifacts under, inside
    --output-dir (spaces/slashes replaced with underscores). Reimplemented
    here rather than imported, since nothing else in this file imports
    Mal-ID-Lite's own Python package directly (everything else goes through
    subprocess calls) -- this one small function is simple enough to keep in
    sync by hand.
    """
    def _safe(s):
        return s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{_safe(disease)}_vs_{_safe(reference)}"


def find_predictions_csv_for_disease(output_dir, disease, healthy_label):
    """Like mal_id_lite_disease_classification.find_predictions_csv(), but
    scoped to one disease's own pair subdirectory. Multi-binary training
    (see run_training()) writes a separate ensemble_predictions.csv per
    disease, each under its own {disease}_vs_{healthy_label} subdirectory of
    --output-dir, rather than one shared file directly under --output-dir the
    way plain binary mode does.
    """
    pair_dir = Path(output_dir) / _make_pair_dir_name(disease, healthy_label)
    matches = sorted(pair_dir.rglob("ensemble_predictions.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No ensemble_predictions.csv found under {pair_dir} for disease "
            f"'{disease}' after multi-binary training. Expected Mal-ID-Lite's "
            f"own pair-subdirectory naming ({_make_pair_dir_name(disease, healthy_label)}) "
            f"under {output_dir} -- if Mal-ID-Lite's own naming convention has "
            f"changed, update _make_pair_dir_name() to match."
        )
    return matches[-1]


def _validate_reused_depth_cache_diseases(cache_dir, diseases, healthy_label, tag):
    """When run_one_depth_repeat() finds an already-built cache for this
    (depth, repeat) tag, verify it actually covers every disease in this run's
    `diseases` list before doing any further work.

    `cache_dir` is scoped only by (depth, repeat), never by which diseases
    were requested -- so if this combination was already cached from an
    earlier run with a *narrower* `diseases` list, and this run asks for a
    *wider* one, the reused cache is missing the newly-added disease's
    participants entirely. Left unchecked, that would surface later as a
    confusing failure (or a silently-incomplete result) deep inside
    Mal-ID-Lite's own multi-binary training step. This is the same
    fail-fast-on-a-mismatched-reused-cache pattern as
    mal_id_lite_disease_classification.py's own
    _validate_reused_cache_metadata(), adapted for a disease *list* instead
    of one target_disease/healthy_label pair.
    """
    cache_dir = Path(cache_dir)
    metadata_processed_path = cache_dir / "metadata_processed.tsv"
    if not metadata_processed_path.exists():
        raise ValueError(
            f"[{tag}] {cache_dir}/metadata.tsv exists but metadata_processed.tsv "
            f"is missing -- an incomplete or corrupted cache build. Delete "
            f"{cache_dir} and re-run to rebuild it from scratch."
        )
    cache_metadata = pd.read_csv(metadata_processed_path, sep="\t")
    if "disease" not in cache_metadata.columns:
        raise ValueError(
            f"[{tag}] {metadata_processed_path} is missing the 'disease' column "
            f"(Mal-ID-Lite's own loader always requires this literal name). "
            f"Available columns: {sorted(cache_metadata.columns)}."
        )
    available_diseases = set(cache_metadata["disease"].unique())
    requested = set(diseases) | {healthy_label}
    missing = requested - available_diseases
    if missing:
        raise ValueError(
            f"[{tag}] this (depth, repeat) combination was already cached, but its "
            f"cache only covers {sorted(available_diseases)} -- missing "
            f"{sorted(missing)} from this run's --target_diseases/--healthy_label "
            f"({sorted(requested)}). This happens when --target_diseases is widened "
            f"after this (depth, repeat) combination was already cached with a "
            f"narrower list. Delete {cache_dir} (or use a fresh --cache_root) to "
            f"rebuild this combination covering the full disease list."
        )


def run_one_depth_repeat(
    depth, repeat_idx, metadata, allowed_specimens, repertoire_data_dir,
    repertoires_index, diseases, dataset_name, cache_root,
    participant_col, disease_col, file_prefix, file_suffix,
    healthy_label, gene_locus, models, model2_abstention_strategy, n_jobs, use_gpu,
):
    """Build one shared cache (+ embeddings) per (depth, repeat), covering
    every disease in `diseases` together, then train them as one multi-binary
    ensemble run -- sharing the expensive, disease-agnostic preprocessing
    (clone_id computation, Stage 1/2, ESM-2 embeddings) across every
    requested disease instead of rebuilding it once per disease. Returns one
    combined standard-schema scores DataFrame covering every disease (each
    row's own `disease_model` column identifies which one it belongs to).

    Caveat: `cache_dir` is scoped only by (depth, repeat) -- via `tag` below
    -- not by which diseases were requested. If this combination's cache was
    already built by an earlier run with a *narrower* `diseases` list, and
    this run asks for a *wider* one, that mismatch is caught up front by
    _validate_reused_depth_cache_diseases() -- a clear ValueError naming the
    missing disease(s) and how to fix it, rather than a confusing failure or
    a silently-incomplete multi-binary training run.
    """
    tag = f"depth{depth}_rep{repeat_idx}"
    scoped = Path(cache_root) / tag
    filtered_dir = scoped / "filtered_repertoires"
    cache_dir = scoped / "cache"
    output_dir = scoped / "output"

    if (cache_dir / "metadata.tsv").exists():
        _validate_reused_depth_cache_diseases(cache_dir, diseases, healthy_label, tag)

    subset_metadata = metadata[metadata["specimen_label"].isin(allowed_specimens)].copy()
    indices_map = build_indices_map(repertoires_index, repeat_idx, depth)

    print(f"  [{tag}] Building depth-filtered repertoires ...")
    build_depth_filtered_repertoires(
        subset_metadata, repertoire_data_dir, indices_map, filtered_dir,
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )

    target_subset = subset_metadata[
        subset_metadata[disease_col].isin([*diseases, healthy_label])
    ].copy()

    print(f"  [{tag}] Consolidating to per-participant files ...")
    has_nt_cdr3 = consolidate_participant_files(
        target_subset, filtered_dir, cache_dir / "consolidated_data",
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )

    staged_metadata_path = cache_dir / "staged_metadata_for_malid_lite.tsv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Mal-ID-Lite's own loader hardcodes the literal "participant_label"/"disease"
    # column names (no way to tell it to use --participant_col/--disease_col's
    # configured names instead), so both get renamed to their canonical names
    # here, right before staging -- same helper and same reasoning as
    # mal_id_lite_disease_classification.py's own staging steps.
    renamed = _rename_for_staging(target_subset, participant_col, disease_col)
    renamed.to_csv(staged_metadata_path, sep="\t", index=False)

    print(f"  [{tag}] Building Mal-ID-Lite cache (skipped if already built) ...")
    build_cache_if_missing(
        cache_dir / "consolidated_data", staged_metadata_path, cache_dir, gene_locus,
        n_jobs, not has_nt_cdr3, force_reprocess=False,
    )

    print(f"  [{tag}] Training (multi-binary: {', '.join(diseases)}) ...")
    run_training(
        cache_dir, f"{dataset_name}_{tag}", gene_locus, diseases, healthy_label,
        models, model2_abstention_strategy, output_dir, n_jobs, use_gpu,
        classification_mode="multi-binary",
    )

    # Multi-binary mode trains each disease as an independent binary ensemble
    # but shares this call's cache/embeddings across all of them (see
    # run_training()'s own docstring). It writes one ensemble_predictions.csv
    # PER disease, under its own output_dir/{disease}_vs_{healthy_label}/
    # subdirectory (see find_predictions_csv_for_disease()) -- read and
    # convert each one separately, then combine.
    all_scores = []
    for disease in diseases:
        predictions_csv = find_predictions_csv_for_disease(output_dir, disease, healthy_label)
        all_scores.append(convert_to_standard_scores(predictions_csv, disease))
    return pd.concat(all_scores, ignore_index=True)


def run_depth_experiment(
    diseases, metadata_path, repertoire_data_dir, depth_indices_path,
    dataset_name, cache_root, output_json=None,
    participant_col="participant_label", disease_col="disease",
    file_prefix="part_table_", file_suffix=".tsv.gz", healthy_label=HEALTHY_LABEL,
    gene_locus="TCR", models=(1, 2, 3), model2_abstention_strategy="fill_models13_mean",
    n_jobs=4, use_gpu=True,
):
    """Run the depth-scaling sweep for every disease in `diseases` together --
    one shared cache/embeddings build per (depth, repeat) covering all of
    them, then one multi-binary training call per (depth, repeat) (see
    run_one_depth_repeat()). AUROC/AUPR/balanced-accuracy/F1 are computed
    separately for each disease -- never blended across diseases sharing the
    same (depth, repeat) run, since they're independent binary classification
    tasks -- so each result entry in the returned/saved list has its own
    "disease" field. "elapsed_seconds" is shared across every disease's
    result for a given (depth, repeat): it's the time for that one combined
    run (sampling + caching + embeddings + multi-binary training), not a
    per-disease time, since that work is no longer done separately per
    disease.
    """
    print(f"Loading depth indices from: {depth_indices_path}")
    index_data = load_depth_indices(depth_indices_path)
    depths = index_data["depths"]
    n_repeats = index_data["n_reps"]
    repertoires_index = index_data["repertoires"]

    print(f"  Depths: {depths}")
    print(f"  Repeats: {n_repeats}")
    print(f"  Repertoires: {len(repertoires_index)}")
    print(f"  Diseases (sampled/cached/embedded together, trained multi-binary): {diseases}")

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
                repertoires_index, diseases, dataset_name, cache_root,
                participant_col, disease_col, file_prefix, file_suffix,
                healthy_label, gene_locus, list(models), model2_abstention_strategy, n_jobs, use_gpu,
            )
            elapsed = time.time() - start_time

            for disease in diseases:
                disease_scores = scores_df[scores_df["disease_model"] == disease]
                y = disease_scores["disease_label"].values
                p = disease_scores["model_score"].values
                preds = (p >= 0.5).astype(int)
                all_results.append({
                    "depth": depth,
                    "repeat": repeat_idx,
                    "disease": disease,
                    "auroc": roc_auc_score(y, p),
                    "aupr": average_precision_score(y, p),
                    "balanced_acc": balanced_accuracy_score(y, preds),
                    "f1": f1_score(y, preds),
                    "n_samples": len(disease_scores),
                    "elapsed_seconds": round(elapsed, 2),
                })

    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "model": "malid_lite",
            "diseases": diseases,
            "depth_indices_path": depth_indices_path,
            "n_repeats": n_repeats,
            "depths": depths,
            "malid_lite_params": {
                "models": list(models),
                "model2_abstention_strategy": model2_abstention_strategy,
                "gene_locus": gene_locus,
                "classification_mode": "multi-binary",
            },
            "results": all_results,
        }
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mal-ID-Lite Sequencing Depth Experiment")
    parser.add_argument("--target_diseases", type=str, nargs="+", required=True,
                        help="Disease(s) to classify (e.g. --target_diseases Lupus HIV). "
                             "All are sampled, cached, and embedded together per "
                             "(depth, repeat) -- not rebuilt once per disease -- then "
                             "trained in a single multi-binary ensemble run "
                             "(--classification-mode multi-binary), each disease vs. "
                             "the shared --healthy_label reference class.")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--repertoire_data_dir", type=str, required=True)
    parser.add_argument("--depth_indices", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset identifier (e.g. malid); used to label the "
                             "Mal-ID-Lite training run for each (depth, repeat) -- not "
                             "disease-specific, since all requested diseases are "
                             "trained together.")
    parser.add_argument("--cache_root", type=str, required=True,
                        help="Root directory for per-(depth,repeat) filtered data/cache/"
                             "embeddings/output (scoped subdirectories are created "
                             "underneath, shared across every --target_diseases value. "
                             "Widening --target_diseases for a (depth, repeat) combination "
                             "already cached here under a narrower list raises a clear "
                             "error rather than training an incomplete ensemble -- delete "
                             "that combination's cache directory, or use a fresh "
                             "--cache_root, to rebuild it.)")
    parser.add_argument("--participant_col", type=str, default="participant_label")
    parser.add_argument("--disease_col", type=str, default="disease")
    parser.add_argument("--file_prefix", type=str, default="part_table_")
    parser.add_argument("--file_suffix", type=str, default=".tsv.gz")
    parser.add_argument("--healthy_label", type=str, default=HEALTHY_LABEL,
                        help="Shared reference/negative class every --target_diseases "
                             "value is trained against.")
    parser.add_argument("--gene_locus", type=str, default="TCR", choices=["TCR"])
    parser.add_argument("--models", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--model2_abstention_strategy", type=str, default="fill_models13_mean")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON. One combined file covering "
                             "every (depth, repeat, disease) combination -- each entry "
                             "in the 'results' list has its own 'disease' field.")
    args = parser.parse_args()

    run_depth_experiment(
        diseases=args.target_diseases,
        metadata_path=args.metadata_path,
        repertoire_data_dir=args.repertoire_data_dir,
        depth_indices_path=args.depth_indices,
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        output_json=args.output_json,
        participant_col=args.participant_col,
        disease_col=args.disease_col,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
        healthy_label=args.healthy_label,
        gene_locus=args.gene_locus,
        models=args.models,
        model2_abstention_strategy=args.model2_abstention_strategy,
        n_jobs=args.n_jobs,
        use_gpu=not args.no_gpu,
    )
