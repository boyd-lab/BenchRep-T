"""
Evaluation script for Mal-ID-Lite disease classification.

Wraps Mal-ID-Lite (models/Mal-ID-Lite, downloaded separately -- see the
README's Setup section), a streamlined
reimplementation of Mal-ID (Zaslavsky et al., Science 2025): a three-model
ensemble (V-J gene usage, convergent CDR3 clusters, ESM-2 sequence
embeddings) combined by a logistic-regression metamodel.

Unlike the other evaluators in this package, Mal-ID-Lite's own
train_ensemble.py handles cross-validation internally (--training-context
cv), so this wrapper's job is purely data-format adaptation between the
driver's staged layout and Mal-ID-Lite's expected layout:

  1. Consolidate the driver's per-specimen repertoire files
     (part_table_{participant}_{specimen}.tsv.gz) into the per-participant
     files Mal-ID-Lite expects (part_table_{participant}.tsv.gz, multiple
     specimens grouped inside via `repertoire_id`). Idempotent -- skipped
     for participants whose consolidated file already exists.
  2. Build Mal-ID-Lite's own preprocessing cache once per dataset (skipped
     if already built), auto-detecting whether the source data has a usable
     nucleotide CDR3 column (`cdr3`) and falling back to --clone-id-use-aa
     when it doesn't (e.g. the immunoSEQ-sourced cohorts, whose nucleotide
     field is a fixed-length read window rather than a clean CDR3 slice).
  3. Invoke train_ensemble.py for the requested --target_disease, reusing
     that cache (so N target diseases in one dataset share one cache/one
     set of ESM-2 embeddings rather than rebuilding per target).
  4. Convert its ensemble_predictions.csv into this benchmark's standard
     scores.csv schema (participant_label, specimen_label, disease_label,
     disease_label_str, method, disease_model, model_score,
     malid_cross_validation_fold_id_when_in_test_set), matching every other
     evaluator's --output_csv.

Like the other disease-classification evaluators, this one accepts
--ext_metadata_path/--ext_data_dir to pool a second cohort into the same
fold-based CV split (utils.cohort_merge), which is how the united
Zaslavsky/Mal-ID + Mitchell T1D evaluation is run. Pooling additionally
canonicalizes V/J gene labels across the two naming conventions and forces
amino-acid clone IDs, since the cohorts disagree on whether a usable
nucleotide CDR3 column exists (see consolidate_participant_files).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils.cohort_adjustments import apply_cohort_adjustment

MAL_ID_LITE_ROOT = Path(__file__).resolve().parent.parent / "models" / "Mal-ID-Lite"
HEALTHY_LABEL = "Healthy/Background"
FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path, sep="\t")


def prepare_metadata_subset(
    metadata, target_disease, disease_col="disease", healthy_label=HEALTHY_LABEL,
    adjust_distribution_by_demographics=False, random_baseline=False, random_baseline_seed=7,
):
    """Filter to target_disease vs. healthy_label and add a binary `label`
    column, then optionally apply the same demographic cohort adjustment
    (utils.cohort_adjustments) every other evaluator in this package
    supports: `adjust_distribution_by_demographics` replaces the healthy pool
    with one matched on the disease's dominant confounder (age or ancestry);
    `random_baseline` additionally resamples healthy uniformly at random to
    the same target N, for the paired random-control baseline.
    """
    mask = metadata[disease_col].isin([target_disease, healthy_label])
    filtered = metadata[mask].copy()
    filtered["label"] = (filtered[disease_col] == target_disease).astype(int)
    if adjust_distribution_by_demographics:
        filtered = apply_cohort_adjustment(
            filtered, target_disease, seed=random_baseline_seed, random_baseline=random_baseline,
        )
    return filtered


def consolidate_participant_files(
    metadata,
    repertoire_data_dir,
    data_dir_out,
    participant_col="participant_label",
    file_prefix="part_table_",
    file_suffix=".tsv.gz",
    canonicalize_genes=False,
):
    """Build one part_table_{participant}{file_suffix} per participant under
    data_dir_out, concatenating that participant's specimen files and setting
    `repertoire_id` = specimen_label on every row. Idempotent: participants
    whose consolidated file already exists are skipped entirely (this is what
    makes repeated calls -- once per target disease in the same dataset --
    cheap after the first).

    Source files are taken from a `file_path` column when the metadata has one
    (as it does after utils.cohort_merge pools two cohorts, each with its own
    directory and filename convention), and otherwise constructed from
    repertoire_data_dir + the prefix/suffix convention.

    `canonicalize_genes` applies utils.gene_harmonization.canonicalize_gene to
    v_call/j_call. Mal-ID-Lite builds its own cache straight from these
    consolidated files, so this is the only point at which the Adaptive-style
    "-1" suffix on IMGT singleton TRBV families can be reconciled; without it a
    pooled run would treat TRBV13 and TRBV13-1 as different genes.

    Returns True only if a genuine nucleotide `cdr3` column is present in
    every cohort, so the caller can decide whether Mal-ID-Lite's default
    clone_id computation (nucleotide-based) will work, or whether
    --clone-id-use-aa is needed. This is probed once per cohort before any
    output is written, rather than taken from whichever file happened to be
    read first: Mal-ID repertoires carry `cdr3` and immunoSEQ-sourced ones do
    not, so a first-file-wins decision would be order-dependent and would leave
    the immunoSEQ half of a pooled run with a half-empty `cdr3` column feeding
    clone assignment. When the cohorts disagree the column is dropped from
    every specimen, so both halves are treated identically.
    """
    data_dir_out = Path(data_dir_out)
    data_dir_out.mkdir(parents=True, exist_ok=True)
    is_gz = file_suffix.endswith(".gz")
    an_existing_out_path = None
    built, skipped, missing = 0, 0, 0

    if canonicalize_genes:
        from utils.gene_harmonization import canonicalize_gene

    # Two source cohorts sharing a participant_label would silently collapse
    # into one consolidated file (and one Mal-ID-Lite "participant"), so refuse
    # rather than corrupt the pooled cohort.
    if "cohort" in metadata.columns and metadata["cohort"].nunique() > 1:
        overlap = set.intersection(*(
            set(part[participant_col]) for _, part in metadata.groupby("cohort")
        ))
        if overlap:
            raise ValueError(
                f"{len(overlap)} participant label(s) appear in more than one "
                f"cohort and would collide during consolidation, e.g. "
                f"{sorted(overlap)[:5]}. Disambiguate them before pooling."
            )

    def source_path(row):
        file_path = row.get("file_path")
        if isinstance(file_path, str) and file_path:
            return Path(file_path)
        return (Path(repertoire_data_dir)
                / f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}")

    # Decide nucleotide-vs-AA clone IDs up front, over one representative
    # source file per cohort, so the answer is fixed before anything is
    # written. Deciding it while writing would be order-dependent: the first
    # participants written would keep `cdr3` and later ones would not.
    probes, seen_cohorts = [], set()
    for _, row in metadata.iterrows():
        cohort = row.get("cohort", "") if "cohort" in metadata.columns else ""
        if cohort in seen_cohorts:
            continue
        src = source_path(row)
        if not src.exists():
            continue
        seen_cohorts.add(cohort)
        probes.append(src)

    has_nt_cdr3 = None
    for src in probes:
        header = pd.read_csv(
            src, sep="\t",
            compression="gzip" if src.name.endswith(".gz") else None, nrows=0,
        )
        found = "cdr3" in header.columns
        has_nt_cdr3 = found if has_nt_cdr3 is None else (has_nt_cdr3 and found)
    drop_nt_cdr3 = has_nt_cdr3 is False

    for participant, group in metadata.groupby(participant_col):
        out_path = data_dir_out / f"{file_prefix}{participant}{file_suffix}"
        if out_path.exists():
            skipped += 1
            an_existing_out_path = out_path
            continue
        frames = []
        for _, row in group.iterrows():
            src = source_path(row)
            if not src.exists():
                missing += 1
                continue
            df = pd.read_csv(
                src, sep="\t",
                compression="gzip" if src.name.endswith(".gz") else None,
                low_memory=False,
            )
            if drop_nt_cdr3 and "cdr3" in df.columns:
                df = df.drop(columns=["cdr3"])
            if canonicalize_genes:
                for col in ("v_call", "j_call"):
                    if col in df.columns:
                        df[col] = df[col].map(canonicalize_gene)
            df["repertoire_id"] = row["specimen_label"]
            frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(
            out_path, sep="\t", index=False, compression="gzip" if is_gz else None
        )
        built += 1

    if has_nt_cdr3 is None and an_existing_out_path is not None:
        # Every participant was already consolidated (idempotent skip), so no
        # source file was probed this call. Check an existing consolidated file
        # instead of silently defaulting to False (which would wrongly force
        # the AA clone_id fallback on every run after the first).
        header = pd.read_csv(
            an_existing_out_path, sep="\t",
            compression="gzip" if is_gz else None, nrows=0,
        )
        has_nt_cdr3 = "cdr3" in header.columns

    if len(probes) > 1 and drop_nt_cdr3:
        print("  Note: source cohorts disagree on the nucleotide `cdr3` column; "
              "dropping it so clone IDs are computed the same way for every "
              "specimen (amino-acid based).")
    print(
        f"  Consolidated participant files: {built} built, {skipped} already present, "
        f"{missing} source specimen files missing."
    )
    return bool(has_nt_cdr3)


def build_cache_if_missing(
    data_dir, metadata_path, cache_dir, gene_locus, n_jobs, use_aa_clone_id, force_reprocess
):
    cache_dir = Path(cache_dir)
    if (cache_dir / "metadata.tsv").exists() and not force_reprocess:
        print(f"  Cache already built at {cache_dir}, skipping cache_and_report_all_data.py.")
        return

    cmd = [
        sys.executable,
        str(MAL_ID_LITE_ROOT / "scripts" / "data" / "cache_and_report_all_data.py"),
        "--data-dir", str(data_dir),
        "--metadata-path", str(metadata_path),
        "--cache-dir", str(cache_dir),
        "--gene-locus", gene_locus,
        "--n-jobs", str(n_jobs),
    ]
    if use_aa_clone_id:
        cmd.append("--clone-id-use-aa")
    if force_reprocess:
        cmd.append("--force-reprocess")

    print(f"  Building cache: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(MAL_ID_LITE_ROOT), check=True)


def run_training(
    cache_dir, dataset_name, gene_locus, target_disease, healthy_label,
    models, model2_abstention_strategy, output_dir, n_jobs, use_gpu, fold_ids=None,
):
    cmd = [
        sys.executable,
        str(MAL_ID_LITE_ROOT / "malid_lite" / "training" / "train_ensemble.py"),
        "--cache-dir", str(cache_dir),
        "--dataset-name", dataset_name,
        "--gene-locus", gene_locus,
        "--classification-mode", "binary",
        "--training-context", "cv",
        "--reference-class", healthy_label,
        "--diseases", target_disease,
        "--models", *[str(m) for m in models],
        "--model2-abstention-strategy", model2_abstention_strategy,
        "--output-dir", str(output_dir),
        "--n-jobs", str(n_jobs),
        "--model3-device", "cuda" if use_gpu else "cpu",
    ]
    if fold_ids is not None:
        cmd += ["--fold-ids", *[str(f) for f in fold_ids]]
    # clone_id parameters are locked in at cache-build time; per Mal-ID-Lite's
    # own guidance, subsequent commands should omit them so the cached values
    # are accepted as-is rather than risking a conflicting-parameter error.
    print(f"  Training: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(MAL_ID_LITE_ROOT), check=True)


def find_predictions_csv(output_dir):
    output_dir = Path(output_dir)
    matches = sorted(output_dir.rglob("ensemble_predictions.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No ensemble_predictions.csv found under {output_dir} after training."
        )
    return matches[-1]


def convert_to_standard_scores(predictions_csv, target_disease):
    df = pd.read_csv(predictions_csv)
    score_col = f"ensemble_P({target_disease})"
    if score_col not in df.columns:
        raise ValueError(
            f"{predictions_csv}: expected column {score_col!r}, have {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=[score_col])
    dropped = before - len(df)
    if dropped:
        print(f"  Note: {dropped} of {before} predictions had no score for "
              f"{score_col!r} (unfilled abstentions) and were dropped.")

    out = pd.DataFrame({
        "participant_label": df["participant_label"],
        "specimen_label": df["specimen_label"],
        "disease_label": (df["true_disease"] == target_disease).astype(int),
        "disease_label_str": df["true_disease"],
        "method": "Mal-ID-Lite",
        "disease_model": target_disease,
        "model_score": df[score_col].astype(float),
        "malid_cross_validation_fold_id_when_in_test_set": df["fold_id"],
    })
    return out


def run_pipeline(
    cache_metadata, repertoire_data_dir, target_disease, dataset_name, cache_dir, model_save_dir,
    participant_col="participant_label", disease_col="disease", file_prefix="part_table_",
    file_suffix=".tsv.gz", healthy_label=HEALTHY_LABEL, gene_locus="TCR", models=(1, 2, 3),
    model2_abstention_strategy="fill_models13_mean", n_jobs=4, use_gpu=True, fold_ids=None,
    force_reprocess=False, ext_metadata_path=None, ext_data_dir=None,
    ext_file_template="{participant_label}_TCRB.tsv", fold_col=FOLD_COL,
):
    """Run the full consolidate -> cache -> train -> convert pipeline and
    return the standard-schema scores DataFrame.

    `cache_metadata` determines both what gets physically consolidated and
    what Mal-ID-Lite's own cache is built from -- pass the full dataset
    metadata (all diseases) when the cache is meant to be reused across
    multiple --target_disease calls sharing one cache_dir (so every
    participant any future target might need already has a file), or a
    single target's (optionally demographically-adjusted) subset when the
    cache is single-use (e.g. one cache_dir per demographic-matched run).
    Training itself always narrows to target_disease vs. healthy_label via
    train_ensemble.py's --diseases flag regardless of which was passed.

    When `ext_metadata_path` is set, a second cohort is pooled into the same
    fold-based CV split via utils.cohort_merge, and V/J gene labels are
    canonicalized across the two naming conventions. A pooled cache is
    inherently single-target (the external cohort is filtered to
    target_disease vs. healthy_label), so it must not be shared with
    single-cohort runs -- give it its own cache_dir.
    """
    if not (MAL_ID_LITE_ROOT / "malid_lite").is_dir():
        raise FileNotFoundError(
            f"Mal-ID-Lite not found at {MAL_ID_LITE_ROOT}. Download it and "
            f"place its contents there -- see the README's Setup section "
            f"(models/Mal-ID-Lite)."
        )

    cache_dir = Path(cache_dir)
    canonicalize_genes = ext_metadata_path is not None

    if ext_metadata_path is not None:
        from utils.cohort_merge import prepare_merged_cohort
        # prepare_merged_cohort resolves external repertoires itself but expects
        # the internal side to already carry file_path, so build it here using
        # the same convention consolidate_participant_files would have used.
        cache_metadata = cache_metadata.copy()
        cache_metadata["file_path"] = [
            str(Path(repertoire_data_dir)
                / f"{file_prefix}{row[participant_col]}_{row['specimen_label']}{file_suffix}")
            for _, row in cache_metadata.iterrows()
        ]
        cache_metadata = prepare_merged_cohort(
            cache_metadata, ext_metadata_path, ext_data_dir, target_disease,
            ext_file_template=ext_file_template, healthy_label=healthy_label,
            fold_col=fold_col, disease_col=disease_col,
        )

    consolidated_dir = cache_dir / "consolidated_data"
    print(f"[1/4] Consolidating per-specimen repertoires into per-participant files "
          f"under {consolidated_dir} ...")
    has_nt_cdr3 = consolidate_participant_files(
        cache_metadata, repertoire_data_dir, consolidated_dir,
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
        canonicalize_genes=canonicalize_genes,
    )
    use_aa_clone_id = not has_nt_cdr3
    print(f"  Nucleotide CDR3 column present: {has_nt_cdr3} "
          f"(clone_id will use {'amino acid' if use_aa_clone_id else 'nucleotide'} sequence).")

    staged_metadata_path = cache_dir / "staged_metadata_for_malid_lite.tsv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    renamed = cache_metadata.rename(columns={disease_col: "disease"}) \
        if disease_col != "disease" else cache_metadata
    # `file_path`, `label`, and `cohort` are this wrapper's own bookkeeping from
    # the merge step; they mean nothing to Mal-ID-Lite's loader, so keep the
    # staged metadata identical in shape to the single-cohort case.
    renamed = renamed.drop(
        columns=[c for c in ("file_path", "label", "cohort") if c in renamed.columns]
    )
    renamed.to_csv(staged_metadata_path, sep="\t", index=False)

    print(f"[2/4] Building Mal-ID-Lite cache at {cache_dir} (skipped if already built) ...")
    build_cache_if_missing(
        consolidated_dir, staged_metadata_path, cache_dir, gene_locus,
        n_jobs, use_aa_clone_id, force_reprocess,
    )

    print(f"[3/4] Training Mal-ID-Lite ensemble for target_disease={target_disease} ...")
    run_training(
        cache_dir, dataset_name, gene_locus, target_disease, healthy_label,
        models, model2_abstention_strategy, model_save_dir, n_jobs, use_gpu, fold_ids=fold_ids,
    )

    print("[4/4] Converting ensemble_predictions.csv to standard scores.csv schema ...")
    predictions_csv = find_predictions_csv(model_save_dir)
    return convert_to_standard_scores(predictions_csv, target_disease)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mal-ID-Lite Disease Classification")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to metadata.tsv")
    parser.add_argument("--repertoire_data_dir", type=str, required=True,
                        help="Directory containing per-specimen repertoire .tsv.gz files")
    parser.add_argument("--target_disease", type=str, required=True,
                        help="Disease to classify (e.g. Lupus, T1D, HIV)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset identifier (e.g. malid, rawat-t1d); used to label "
                             "and namespace the Mal-ID-Lite cache/output.")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Dataset-level Mal-ID-Lite cache directory, shared across "
                             "every --target_disease run for this dataset. Built once, "
                             "reused thereafter (must be the same path across those runs).")
    parser.add_argument("--participant_col", type=str, default="participant_label")
    parser.add_argument("--disease_col", type=str, default="disease")
    parser.add_argument("--fold_col", type=str, default=FOLD_COL)
    parser.add_argument("--file_prefix", type=str, default="part_table_")
    parser.add_argument("--file_suffix", type=str, default=".tsv.gz")
    parser.add_argument("--healthy_label", type=str, default=HEALTHY_LABEL,
                        help="Negative-class / reference label in the disease column.")
    parser.add_argument("--gene_locus", type=str, default="TCR", choices=["TCR"])
    parser.add_argument("--models", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3],
                        help="Which of Mal-ID-Lite's 3 base models to include (default: 1 2 3).")
    parser.add_argument("--model2_abstention_strategy", type=str, default="fill_models13_mean",
                        help="How Mal-ID-Lite handles Model 2 abstentions (default: "
                             "fill_models13_mean, so every specimen receives a score).")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU for Model 3's ESM-2 embedding computation.")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Rebuild the Mal-ID-Lite cache from scratch even if present.")
    parser.add_argument("--model_save_dir", type=str, default=None,
                        help="Directory for Mal-ID-Lite's own trained model / run artifacts "
                             "(passed through as train_ensemble.py's --output-dir).")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save per-sample scores CSV (standard schema).")
    parser.add_argument("--adjust_distribution_by_demographics", action="store_true",
                        help="Apply per-disease cohort distribution adjustment (see "
                             "utils.cohort_adjustments): HIV filters to African ancestry; "
                             "Lupus/T1D/Influenza/Covid19 subsample Healthy/Background to "
                             "match the disease cohort's age distribution.")
    parser.add_argument("--random_baseline_seeds", type=int, nargs="+", default=None,
                        help="Run the random-sampling healthy baseline for each seed "
                             "(implies --adjust_distribution_by_demographics). For each "
                             "seed, healthy is resampled uniformly at random to the same "
                             "target N as the demographic-matched cohort. Each seed gets "
                             "its own cache/output subdirectory (the resampled cohort "
                             "differs per seed, so nothing is reused across them); results "
                             "from all seeds are concatenated with a `random_baseline_seed` "
                             "column. Example: 7 14 21 28 35.")
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limit training to fold IDs 0..max_folds-1 (passed through as "
                             "train_ensemble.py's --fold-ids). Default: all folds.")
    parser.add_argument("--ext_metadata_path", type=str, default=None,
                        help="Optional external-cohort metadata TSV (MAL-ID column style). "
                             "When set, external samples are pooled into the same fold-based "
                             "CV split as the internal cohort and V/J genes are canonicalized "
                             "-- this is how the united Zaslavsky/Mal-ID + Mitchell T1D "
                             "evaluation is run. Give a pooled run its own --cache_dir; the "
                             "cache is single-target and must not be shared with "
                             "single-cohort runs.")
    parser.add_argument("--ext_data_dir", type=str, default=None,
                        help="Directory containing the external cohort repertoire files "
                             "(required when --ext_metadata_path is provided).")
    parser.add_argument("--ext_file_template", type=str,
                        default="{participant_label}_TCRB.tsv",
                        help="Filename template for external repertoires. For a cohort "
                             "staged by utils.stage_disease_data, use "
                             "'part_table_{participant_label}_{specimen_label}.tsv.gz'.")
    args = parser.parse_args()

    if args.model_save_dir is None:
        parser.error("--model_save_dir is required (used as Mal-ID-Lite's training output dir)")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max_folds must be >= 1")
    if args.ext_metadata_path is not None and args.ext_data_dir is None:
        parser.error("--ext_data_dir is required when --ext_metadata_path is provided")
    fold_ids = list(range(args.max_folds)) if args.max_folds is not None else None

    full_metadata = load_metadata(args.metadata_path)
    # utils.stage_disease_data's uniform staging view adds the canonical
    # fold-ID column (args.fold_col) while preserving each cohort's native
    # fold column. For cohorts whose native column has its own name --
    # currently "CV_fold" for rawat-t1d, tb, and ra -- both end up in the
    # same staged metadata.tsv. Mal-ID-Lite's own loader (normalize_fold_column)
    # rejects having more than one fold-ID column present, so drop the native
    # one here now that the canonical one has been read.
    if "CV_fold" in full_metadata.columns and "CV_fold" != args.fold_col:
        full_metadata = full_metadata.drop(columns=["CV_fold"])
    pipeline_kwargs = dict(
        repertoire_data_dir=args.repertoire_data_dir,
        target_disease=args.target_disease,
        dataset_name=args.dataset_name,
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
        fold_ids=fold_ids,
        force_reprocess=args.force_reprocess,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_file_template=args.ext_file_template,
        fold_col=args.fold_col,
    )

    if args.random_baseline_seeds:
        seed_dfs = []
        for seed in args.random_baseline_seeds:
            print(f"\n{'#'*60}")
            print(f"# RANDOM BASELINE RUN — seed={seed}")
            print(f"{'#'*60}")
            metadata_subset = prepare_metadata_subset(
                full_metadata, args.target_disease, args.disease_col, args.healthy_label,
                adjust_distribution_by_demographics=True, random_baseline=True,
                random_baseline_seed=seed,
            )
            seed_df = run_pipeline(
                cache_metadata=metadata_subset,
                cache_dir=Path(args.cache_dir) / f"seed{seed}",
                model_save_dir=Path(args.model_save_dir) / f"seed{seed}",
                **pipeline_kwargs,
            )
            seed_df["random_baseline_seed"] = seed
            seed_dfs.append(seed_df)
        scores_df = pd.concat(seed_dfs, axis=0, ignore_index=True)
    elif args.adjust_distribution_by_demographics:
        metadata_subset = prepare_metadata_subset(
            full_metadata, args.target_disease, args.disease_col, args.healthy_label,
            adjust_distribution_by_demographics=True,
        )
        scores_df = run_pipeline(
            cache_metadata=metadata_subset,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )
    elif args.ext_metadata_path is not None:
        # Pooling narrows the external cohort to target_disease vs. healthy, so
        # the cache is single-target either way; filter the internal side to
        # match rather than consolidating diseases this run will never train on.
        metadata_subset = prepare_metadata_subset(
            full_metadata, args.target_disease, args.disease_col, args.healthy_label,
        )
        scores_df = run_pipeline(
            cache_metadata=metadata_subset,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )
    else:
        # No demographic adjustment: cache_metadata is the full, unfiltered
        # dataset metadata (every disease, not just this target) so that if
        # --cache_dir is reused across multiple --target_disease calls (the
        # normal disease-classification driver pattern), every participant a
        # later target might need already has a consolidated file, rather
        # than only whichever target happened to run first.
        scores_df = run_pipeline(
            cache_metadata=full_metadata,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
