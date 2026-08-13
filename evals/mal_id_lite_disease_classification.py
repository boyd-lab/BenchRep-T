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
--ext_metadata_path/--ext_data_dir/--ext_cache_dir to pool a second cohort
into the same fold-based CV split, which is how the united Zaslavsky/Mal-ID
+ Mitchell T1D evaluation is run. Unlike the other evaluators, pooling here
never merges raw repertoire files: each cohort's own Mal-ID-Lite cache is
resolved independently (reused if already built, or built from scratch --
cache and ESM-2 embeddings only, never trained on standalone) using the
exact same code path a normal standalone run of that cohort already uses,
and only the two finished caches are merged, via Mal-ID-Lite's own
scripts/cache_tools/merge_caches.py. This is what actually reproduces the
manuscript's methodology: each cohort keeps its own clone_id basis (malid's
own nucleotide-based clone_id, Mitchell's own amino-acid-based one) rather
than forcing both onto one uniform basis, and no gene-label canonicalization
is needed at all, since Mal-ID-Lite's own Stage-1 preprocessing already
normalizes every cohort's v_call/j_call to v_gene/j_gene independently.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cohort_adjustments import apply_cohort_adjustment

MAL_ID_LITE_ROOT = Path(__file__).resolve().parent.parent / "models" / "Mal-ID-Lite"
HEALTHY_LABEL = "Healthy/Background"
FOLD_COL = "malid_cross_validation_fold_id_when_in_test_set"


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path, sep="\t")


def _rename_disease_col_back(metadata, disease_col):
    """A cache's own metadata_processed.tsv always uses Mal-ID-Lite's
    canonical "disease" column name (see the disease_col -> "disease" rename
    in resolve_cache()'s/run_pipeline()'s own staging step, applied before
    Mal-ID-Lite ever sees a file). Rename it back to whatever --disease_col
    this run actually expects, so every caller can rely on that column being
    named disease_col, regardless of whether the metadata came from a fresh
    raw file or an existing cache. No-op when disease_col is already
    "disease" (true for every invocation in this codebase today, since
    --disease_col is never actually overridden from its default).
    """
    if disease_col != "disease" and "disease" in metadata.columns:
        return metadata.rename(columns={"disease": disease_col})
    return metadata


def _rename_participant_col_back(metadata, participant_col):
    """Rename Mal-ID-Lite's own canonical "participant_label" column back to
    whatever --participant_col this run expects. Mirrors
    _rename_disease_col_back above, with one crucial difference: unlike
    disease_col, prepare_merged_cohort() and merge_caches.py both hardcode
    the literal "participant_label" name with no way to override it (see
    resolve_cache()'s docstring), so this must NEVER be applied to metadata
    destined for either of those two pooling calls -- only to metadata
    destined for this wrapper's own downstream use (e.g.
    consolidate_participant_files(), which does expect participant_col's own
    configured name). No-op when participant_col is already
    "participant_label" (true for every invocation in this codebase today).
    """
    if participant_col != "participant_label" and "participant_label" in metadata.columns:
        return metadata.rename(columns={"participant_label": participant_col})
    return metadata


def _rename_for_staging(metadata, participant_col, disease_col):
    """Rename participant_col/disease_col to Mal-ID-Lite's own hardcoded
    canonical names ("participant_label"/"disease") before staging a file for
    Mal-ID-Lite's own tools (cache_and_report_all_data.py,
    compute_model3_embeddings.py) or merge_caches.py to read. Neither
    supports a customizable name for either column: both are literal,
    hardcoded strings in Mal-ID-Lite's own loader
    (malid_lite/dataloader/mal_id_published.py's own required_metadata_cols)
    and in merge_caches.py's own REQUIRED_METADATA_COLS. No-op for any column
    already at its canonical name (true for every invocation in this
    codebase today, since neither --participant_col nor --disease_col is
    ever actually overridden from its default).
    """
    rename_map = {}
    if participant_col != "participant_label":
        rename_map[participant_col] = "participant_label"
    if disease_col != "disease":
        rename_map[disease_col] = "disease"
    return metadata.rename(columns=rename_map) if rename_map else metadata


def load_or_read_cached_metadata(metadata_path, cache_dir, fold_col, disease_col, participant_col):
    """Resolve a cohort's own metadata: load fresh from `metadata_path` if
    given, or read back an already-built `cache_dir`'s own
    metadata_processed.tsv otherwise.

    Used both directly (by the plain, non-pooled branches in __main__, which
    already have a concrete --cache_dir to fall back on) and as the first
    step of resolve_cache() below (which additionally builds a fresh cache
    when there is nothing to read back yet).

    When reading from cache, renames Mal-ID-Lite's own canonical
    "participant_label"/"disease" columns back to whatever
    --participant_col/--disease_col this run was given, so callers can treat
    this function's return value uniformly regardless of whether the
    metadata came from a fresh raw file (already using this run's own
    naming) or an existing cache (always canonical). NOT used for
    _run_pooled_pipeline()'s ext_metadata_for_merge, which deliberately keeps
    "participant_label" canonical instead -- see the comment there.
    """
    if metadata_path is not None:
        metadata = load_metadata(metadata_path)
        # utils.stage_disease_data's uniform staging view can carry both the
        # canonical fold-ID column (fold_col) and a cohort's own native one
        # (e.g. "CV_fold" for rawat-t1d/tb/ra). Mal-ID-Lite's own loader
        # (normalize_fold_column) rejects having more than one fold-ID
        # column present, so drop the native one now that the canonical one
        # has been read. Not needed when reading back an already-built
        # cache's own metadata_processed.tsv below: that file was produced
        # by this exact same drop already happening at whichever earlier run
        # built the cache.
        if "CV_fold" in metadata.columns and "CV_fold" != fold_col:
            metadata = metadata.drop(columns=["CV_fold"])
        return metadata

    metadata_processed_path = Path(cache_dir) / "metadata_processed.tsv"
    if not metadata_processed_path.exists():
        raise FileNotFoundError(
            f"metadata_path was not given and no cache exists yet at {cache_dir} "
            f"(expected {metadata_processed_path}). This should already have "
            f"been caught by argument validation at startup -- if you're seeing "
            f"this, pass --metadata_path (and --repertoire_data_dir) to build "
            f"the cache first."
        )
    metadata = pd.read_csv(metadata_processed_path, sep="\t")
    metadata = _rename_disease_col_back(metadata, disease_col)
    return _rename_participant_col_back(metadata, participant_col)


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
    (as it does when resolve_cache() pre-resolves it via `file_path_template`,
    for an external cohort whose raw files don't follow the fixed
    {file_prefix}{participant}_{specimen}{file_suffix} convention), and
    otherwise constructed from repertoire_data_dir + the prefix/suffix
    convention.

    `canonicalize_genes` applies utils.gene_harmonization.canonicalize_gene to
    v_call/j_call, collapsing the Adaptive-style "-1" suffix on IMGT singleton
    TRBV families. Unlike several other evaluators in this package, a pooled
    Mal-ID-Lite run never needs this: Mal-ID-Lite's own Stage-1 preprocessing
    normalizes every cohort's v_call/j_call to v_gene/j_gene independently of
    the others (see this module's own docstring), so no caller in this file
    passes True. Kept only in case a future caller needs it.

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
        cmd.append("--force-clone-id")
    if force_reprocess:
        cmd.append("--force-reprocess")

    print(f"  Building cache: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(MAL_ID_LITE_ROOT), check=True)


def _validate_reused_cache_metadata(
    cache_metadata, target_disease, healthy_label, fold_col, side_label,
):
    """Sanity-check an already-built, reused cache's own metadata against the
    arguments this run was invoked with, before spending any time on the rest
    of the pipeline -- catches a stale or wrong --cache_dir/--ext_cache_dir
    pointed at by mistake, with a clear message, instead of a confusing
    failure deep inside Mal-ID-Lite's own training code (or, worse, a
    silently-empty pooled cohort).

    Checks the literal column names "participant_label" and "disease"
    (Mal-ID-Lite's own canonical, non-customizable names -- both are
    hardcoded literal strings in Mal-ID-Lite's own loader and in
    merge_caches.py's own required-columns check, with no argument anywhere
    to override either), not whatever --participant_col/--disease_col this
    run's raw metadata happens to use: resolve_cache() below always renames
    both to their canonical names before Mal-ID-Lite (or merge_caches.py)
    ever sees a staged file, so *any* previously-built cache's own
    metadata_processed.tsv already has those canonical names baked in,
    regardless of which --participant_col/--disease_col values the run that
    originally built it was given. --fold_col is never renamed this way
    (Mal-ID-Lite's own loader accepts either "CV_fold" or the legacy name,
    and --fold_col is only ever set to one of those two), so it's checked
    using this run's own argument value directly.
    """
    if fold_col not in cache_metadata.columns:
        raise ValueError(
            f"{side_label} cache's metadata_processed.tsv is missing column "
            f"'{fold_col}' (from --fold_col). Available columns: "
            f"{sorted(cache_metadata.columns)}."
        )
    for canonical_name in ("participant_label", "disease"):
        if canonical_name not in cache_metadata.columns:
            raise ValueError(
                f"{side_label} cache's metadata_processed.tsv is missing the "
                f"'{canonical_name}' column (Mal-ID-Lite's own loader always "
                f"requires this literal name, regardless of --participant_col/"
                f"--disease_col). Available columns: {sorted(cache_metadata.columns)}."
            )
    available_diseases = set(cache_metadata["disease"].unique())
    for arg_name, value in (("--target_disease", target_disease), ("--healthy_label", healthy_label)):
        if value not in available_diseases:
            raise ValueError(
                f"{arg_name}={value!r} not found in the {side_label} cache's 'disease' "
                f"column. Available values: {sorted(available_diseases)}. Either this is "
                f"the wrong cache, or {arg_name} has a typo."
            )


def _cross_check_raw_vs_cache_metadata(
    raw_metadata, cache_metadata, participant_col, disease_col, fold_col, side_label,
):
    """When both raw metadata and an already-built cache are given for the
    same side, verify they actually agree -- participant set, disease label,
    and fold assignment -- rather than silently preferring the cache. Catches,
    e.g., pointing --metadata_path at updated data while --cache_dir/
    --ext_cache_dir still holds a stale build.

    Compares `raw_metadata`'s own `participant_col`/`disease_col` against the
    cache's canonical "participant_label"/"disease" columns (see
    _validate_reused_cache_metadata for why these can differ in name while
    meaning the same thing).
    """
    raw_subset = (
        raw_metadata[[participant_col, disease_col, fold_col]]
        .drop_duplicates(subset=participant_col)
        .set_index(participant_col)
    )
    cache_subset = (
        cache_metadata[["participant_label", "disease", fold_col]]
        .drop_duplicates(subset="participant_label")
        .set_index("participant_label")
    )
    common = raw_subset.index.intersection(cache_subset.index)
    n_mismatched = sum(
        1 for pid in common
        if raw_subset.loc[pid, disease_col] != cache_subset.loc[pid, "disease"]
        or raw_subset.loc[pid, fold_col] != cache_subset.loc[pid, fold_col]
    )
    n_only_in_raw = len(raw_subset.index.difference(cache_subset.index))
    n_only_in_cache = len(cache_subset.index.difference(raw_subset.index))
    if n_mismatched or n_only_in_raw or n_only_in_cache:
        raise ValueError(
            f"{side_label}: raw metadata was given alongside an already-built cache, "
            f"but they disagree: {n_mismatched} participant(s) have a different "
            f"disease/fold assignment, {n_only_in_raw} participant(s) only in the raw "
            f"metadata, {n_only_in_cache} participant(s) only in the cache. Use "
            f"--force_reprocess to rebuild this side from the raw metadata (note: for a "
            f"pooled run, --force_reprocess only rebuilds the *merge*, never a source "
            f"side -- delete this side's cache directory to force a full rebuild), or "
            f"omit the raw metadata/repertoire-data-dir arguments to just use the "
            f"existing cache as-is."
        )


def _compute_embeddings_if_needed(cache_dir, metadata_path, gene_locus, dataset_name, side_label):
    """Invoke compute_model3_embeddings.py against `cache_dir`. Safe and cheap
    to call unconditionally, including on an already-fully-embedded cache:
    the script has its own built-in resume logic (skips any participant whose
    3 embedding files already exist and pass a shape/consistency check), and
    if every participant is already done, it returns before ever loading the
    ESM-2 model -- just a fast per-participant file/shape scan, not a
    recompute.

    Calling this on a *reused* cache (not just a freshly-built one) is what
    catches a cache_dir that was reused from a prior run that skipped Model 3
    (--models 1 2) or whose embeddings computation was interrupted partway
    through, before merge_caches.py would otherwise fail on it later with a
    much less informative "missing files" error, deep inside a different
    script this side effectively never ran.
    """
    print(f"  {side_label}: checking/computing ESM-2 embeddings for every participant "
          f"(this never happens automatically here the way it does during training, "
          f"since this side is never trained on standalone; cheap no-op if already "
          f"complete) ...")
    embed_cmd = [
        sys.executable,
        str(MAL_ID_LITE_ROOT / "malid_lite" / "training" / "compute_model3_embeddings.py"),
        "--metadata-path", str(metadata_path),
        "--cache-dir", str(cache_dir),
        "--gene-locus", gene_locus,
        "--dataset-name", dataset_name,
    ]
    print(f"  {' '.join(embed_cmd)}")
    subprocess.run(embed_cmd, cwd=str(MAL_ID_LITE_ROOT), check=True)


def resolve_cache(
    cache_dir, metadata_path, repertoire_data_dir, dataset_name, gene_locus, n_jobs,
    participant_col, disease_col, fold_col, file_prefix, file_suffix,
    target_disease, healthy_label, compute_embeddings, side_label,
    file_path_template=None,
):
    """Resolve one cohort's own Mal-ID-Lite cache: reuse it as-is if
    `cache_dir` already contains a built cache (`metadata.tsv` present), or
    build one from scratch -- participant + fold cache, and, if
    `compute_embeddings`, ESM-2 embeddings too -- from `metadata_path`/
    `repertoire_data_dir` if not. Building never trains a model on this
    cohort; that always happens separately, afterward, on whichever cache the
    caller actually wants to train on.

    When `compute_embeddings`, embeddings are checked/computed on *both*
    branches, not just when building fresh: compute_model3_embeddings.py's
    own resume logic makes this a cheap no-op on an already-fully-embedded
    reused cache (see _compute_embeddings_if_needed), and it's what catches a
    reused cache_dir whose embeddings are missing or incomplete (e.g. from a
    prior run that skipped Model 3, or that got interrupted mid-computation)
    before merge_caches.py would otherwise fail on it later, less clearly.

    `file_path_template` is used only when building a fresh cache, and only
    for a cohort whose raw repertoire files don't follow the fixed
    {file_prefix}{participant}_{specimen}{file_suffix} convention
    consolidate_participant_files() otherwise assumes (that convention is
    specific to files already staged by utils.stage_disease_data). Pass a
    format-string template instead (e.g. "{participant_label}_TCRB.tsv",
    Mitchell T1D's own raw file naming, the same string the old raw-file-merge
    approach's --ext_file_template used) to resolve each participant's actual
    file explicitly, reusing utils.cohort_merge's own path-resolution logic
    (including its DenverT1D zero-padding special case) -- so this finds
    exactly the same files the old approach would have found for the same
    cohort. Leave as None (the internal side, and any external cohort already
    staged into the fixed convention) to use consolidate_participant_files()'s
    own fallback instead.

    This is the one place cache resolution happens for every case this script
    supports:
      - The plain, single-cohort case (`compute_embeddings=False`): a normal
        training run immediately afterward will auto-compute-and-cache any
        embeddings it needs as a side effect anyway (see
        malid_lite/training/train_model3.py's own docstring), so computing
        them explicitly here would just be redundant work.
      - A pooled run's internal and external sides, resolved independently
        (`compute_embeddings=True`): pooling deliberately never trains a side
        standalone (training happens exactly once, on the merged result), so
        nothing else would ever trigger that same auto-compute -- it has to
        happen explicitly here instead.
    Building each side through this exact same code path, independently, is
    also what lets a pooled run reproduce the actual published methodology
    automatically: whichever cohort has a genuine nucleotide `cdr3` column
    ends up on nucleotide-based clone_id, whichever doesn't ends up on
    amino-acid-based, with no cross-cohort comparison or special-casing.

    Returns the resolved metadata for this cohort as a DataFrame: read back
    from the cache's own metadata_processed.tsv (whether just built, or
    already present beforehand) -- never `metadata_path`'s own contents
    as-is, since Mal-ID-Lite's own cache-building step can drop participants
    that fail its QC, and the cache's own file is the authoritative record of
    who actually made it into the cache. The returned DataFrame's disease
    column is renamed back to --disease_col, but its participant column is
    deliberately left as the canonical "participant_label" (never renamed to
    --participant_col): unlike disease_col, prepare_merged_cohort() and
    merge_caches.py -- the only consumers of this return value, both used
    only for pooling -- hardcode that literal name with no way to override
    it, so this function must hand it back unchanged for pooling to work
    regardless of --participant_col.
    """
    cache_dir = Path(cache_dir)
    cache_already_built = (cache_dir / "metadata.tsv").exists()

    if cache_already_built:
        cache_metadata = pd.read_csv(cache_dir / "metadata_processed.tsv", sep="\t")
        if metadata_path is not None:
            raw_metadata = load_or_read_cached_metadata(
                metadata_path, cache_dir, fold_col, disease_col, participant_col,
            )
            _cross_check_raw_vs_cache_metadata(
                raw_metadata, cache_metadata, participant_col, disease_col, fold_col, side_label,
            )
        _validate_reused_cache_metadata(
            cache_metadata, target_disease, healthy_label, fold_col, side_label,
        )
        print(f"  {side_label}: cache already built at {cache_dir}, reusing as-is.")
        if compute_embeddings:
            _compute_embeddings_if_needed(
                cache_dir, cache_dir / "metadata_processed.tsv", gene_locus, dataset_name, side_label,
            )
        return _rename_disease_col_back(cache_metadata, disease_col)

    if metadata_path is None:
        # Argument validation in __main__ should already have required
        # metadata_path/repertoire_data_dir whenever a side's cache doesn't
        # already exist, so reaching here means this function was called
        # directly with inconsistent arguments, not a normal user mistake.
        raise ValueError(
            f"No cache found at {cache_dir} for the {side_label} cohort, and no "
            f"metadata/repertoire-data-dir was given to build one."
        )

    raw_metadata = load_or_read_cached_metadata(
        metadata_path, cache_dir, fold_col, disease_col, participant_col,
    )

    if file_path_template is not None:
        # consolidate_participant_files() only knows the fixed
        # {file_prefix}{participant}_{specimen}{file_suffix} convention (via
        # its own source_path() fallback) unless a row already has a
        # `file_path` set. Pre-compute one explicitly here using this
        # cohort's own (possibly different) raw filename convention, exactly
        # as the old raw-file-merge approach's prepare_merged_cohort() did
        # for the external side.
        from utils.cohort_merge import _resolve_external_path
        raw_metadata = raw_metadata.copy()
        raw_metadata["file_path"] = raw_metadata.apply(
            lambda row: _resolve_external_path(row, repertoire_data_dir, file_path_template),
            axis=1,
        )

    consolidated_dir = cache_dir / "consolidated_data"
    print(f"  {side_label}: no cache yet at {cache_dir} -- building one from scratch. "
          f"Consolidating repertoires under {consolidated_dir} ...")
    has_nt_cdr3 = consolidate_participant_files(
        raw_metadata, repertoire_data_dir, consolidated_dir,
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )
    use_aa_clone_id = not has_nt_cdr3
    print(f"  {side_label}: nucleotide CDR3 column present: {has_nt_cdr3} "
          f"(clone_id will use {'amino acid' if use_aa_clone_id else 'nucleotide'} sequence).")

    staged_metadata_path = cache_dir / "staged_metadata_for_malid_lite.tsv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Pass this cohort's *full* metadata (every disease, not just
    # target_disease/healthy_label) to the cache build below, even though
    # this run only cares about one target_disease. Building narrower would
    # still leave cache_dir/metadata.tsv behind, so a later run reusing this
    # same cache_dir for a *different* disease would see "already built" and
    # skip rebuilding -- even though it's actually missing every participant
    # outside this run's own target_disease/healthy_label. Using the full
    # metadata here avoids that trap and matches how a normal standalone run
    # already treats a --cache_dir meant to be reused across multiple
    # --target_disease calls.
    renamed = _rename_for_staging(raw_metadata, participant_col, disease_col)
    # `file_path` (set above, only when file_path_template was given) is this
    # function's own bookkeeping for consolidate_participant_files() and
    # means nothing to Mal-ID-Lite's loader.
    renamed = renamed.drop(columns=[c for c in ("file_path",) if c in renamed.columns])
    renamed.to_csv(staged_metadata_path, sep="\t", index=False)

    # force_reprocess is intentionally never threaded through to this call:
    # for a pooled run, --force_reprocess only ever means "redo the merge
    # into --pooled_cache_dir," never "rebuild this side's own cache" (if
    # this side's cache already existed, the branch above already returned
    # before reaching here). Rebuilding a source side is only ever triggered
    # by there being no cache there at all yet.
    build_cache_if_missing(
        consolidated_dir, staged_metadata_path, cache_dir, gene_locus, n_jobs,
        use_aa_clone_id, force_reprocess=False,
    )

    if compute_embeddings:
        _compute_embeddings_if_needed(
            cache_dir, staged_metadata_path, gene_locus, dataset_name, side_label,
        )

    built_metadata = pd.read_csv(cache_dir / "metadata_processed.tsv", sep="\t")
    return _rename_disease_col_back(built_metadata, disease_col)


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


def _run_pooled_pipeline(
    cache_dir, metadata_path, repertoire_data_dir, dataset_name,
    ext_metadata_path, ext_data_dir, ext_cache_dir, ext_dataset_name, ext_file_template,
    pooled_cache_dir, pooled_dataset_name,
    target_disease, healthy_label, participant_col, disease_col, fold_col,
    file_prefix, file_suffix, gene_locus, n_jobs, models, model2_abstention_strategy,
    model_save_dir, use_gpu, fold_ids, force_reprocess,
):
    """Pooled-run pipeline: resolve each cohort's own cache independently
    (reusing it if already built, or building it from scratch -- cache and
    ESM-2 embeddings only, never training), merge the two finished caches via
    Mal-ID-Lite's own scripts/cache_tools/merge_caches.py, then train exactly
    once on the merged result.

    Unlike the raw-file-merge approach this replaces, neither cohort's raw
    repertoire files are ever combined, and neither cohort's clone_id basis
    is forced to match the other's: each side is built through the exact
    same code path a normal standalone run already uses (see resolve_cache),
    so malid naturally keeps its own nucleotide-based clone_id and Mitchell
    its own amino-acid-based one, exactly as in the actual published result.
    """
    from utils.cohort_merge import prepare_merged_cohort

    print(f"[1/5] Resolving internal cohort's own cache at {cache_dir} ...")
    internal_metadata = resolve_cache(
        cache_dir=cache_dir, metadata_path=metadata_path, repertoire_data_dir=repertoire_data_dir,
        dataset_name=dataset_name, gene_locus=gene_locus, n_jobs=n_jobs,
        participant_col=participant_col, disease_col=disease_col, fold_col=fold_col,
        file_prefix=file_prefix, file_suffix=file_suffix,
        target_disease=target_disease, healthy_label=healthy_label,
        compute_embeddings=True, side_label="internal",
    )

    print(f"[2/5] Resolving external cohort's own cache at {ext_cache_dir} ...")
    resolve_cache(
        cache_dir=ext_cache_dir, metadata_path=ext_metadata_path, repertoire_data_dir=ext_data_dir,
        dataset_name=ext_dataset_name, gene_locus=gene_locus, n_jobs=n_jobs,
        participant_col=participant_col, disease_col=disease_col, fold_col=fold_col,
        file_prefix=file_prefix, file_suffix=file_suffix,
        target_disease=target_disease, healthy_label=healthy_label,
        compute_embeddings=True, side_label="external",
        file_path_template=ext_file_template,
    )
    pooled_cache_dir = Path(pooled_cache_dir)
    pooled_cache_dir.mkdir(parents=True, exist_ok=True)
    # utils.cohort_merge.prepare_merged_cohort() reads the external metadata
    # itself from a path, rather than taking a pre-loaded DataFrame (see that
    # module), and looks for a disease column literally named disease_col --
    # but the external cache's own metadata_processed.tsv always uses
    # Mal-ID-Lite's canonical "disease" name (see resolve_cache()). Reconcile
    # the two via the same disease-only helper resolve_cache() uses, and
    # write the result to a scratch file for prepare_merged_cohort() to read,
    # since it takes a path, not a DataFrame. Deliberately NOT
    # load_or_read_cached_metadata(): that function also renames
    # "participant_label" back to --participant_col, which would break
    # prepare_merged_cohort()'s own hardcoded requirement for the literal
    # "participant_label" name (see resolve_cache()'s docstring) -- this
    # scratch file must keep that name canonical. ext_data_dir=None tells it
    # to skip raw-repertoire-file resolution entirely: with the raw-file
    # merge approach removed, nothing downstream ever reads a raw external
    # file by path, so there's nothing to resolve and nothing that needs to
    # exist on disk beyond the two caches being merged below.
    ext_metadata_for_merge = pd.read_csv(Path(ext_cache_dir) / "metadata_processed.tsv", sep="\t")
    ext_metadata_for_merge = _rename_disease_col_back(ext_metadata_for_merge, disease_col)
    resolved_ext_metadata_path = pooled_cache_dir / "_ext_metadata_for_merge.tsv"
    ext_metadata_for_merge.to_csv(resolved_ext_metadata_path, sep="\t", index=False)

    print("[3/5] Merging internal + external metadata for the pooled CV split ...")
    internal_prepared = prepare_metadata_subset(
        internal_metadata, target_disease, disease_col, healthy_label,
    )
    merged_metadata = prepare_merged_cohort(
        internal_prepared, str(resolved_ext_metadata_path), None, target_disease,
        healthy_label=healthy_label, fold_col=fold_col, disease_col=disease_col,
    )
    # prepare_merged_cohort() only ever tags each row 'internal'/'external'
    # (the `cohort` column) -- it never records which *named* cohort a row
    # actually came from. Add that here so the merged cache's own metadata
    # is self-describing to a human browsing it later.
    merged_metadata["source_dataset"] = np.where(
        merged_metadata["cohort"] == "internal", dataset_name, ext_dataset_name,
    )

    staged_metadata_path = pooled_cache_dir / "staged_metadata_for_malid_lite.tsv"
    renamed = merged_metadata.rename(columns={disease_col: "disease"}) \
        if disease_col != "disease" else merged_metadata
    # `file_path` and `label` are this wrapper's/prepare_merged_cohort()'s own
    # bookkeeping and mean nothing to Mal-ID-Lite's loader. `cohort` is kept
    # (unlike the single-cohort case below) -- merge_caches.py, invoked below
    # on this exact file, requires it as its own --origin-column to know
    # which of the two source caches each participant's files come from;
    # dropping it here would make that call fail with "origin column
    # 'cohort' not found."
    renamed = renamed.drop(columns=[c for c in ("file_path", "label") if c in renamed.columns])
    renamed.to_csv(staged_metadata_path, sep="\t", index=False)

    if (pooled_cache_dir / "metadata.tsv").exists() and not force_reprocess:
        print(f"[4/5] Pooled cache already merged at {pooled_cache_dir}, skipping merge_caches.py.")
    else:
        print(f"[4/5] Merging the two resolved caches into {pooled_cache_dir} ...")
        merge_cmd = [
            sys.executable,
            str(MAL_ID_LITE_ROOT / "scripts" / "cache_tools" / "merge_caches.py"),
            "--metadata", str(staged_metadata_path),
            "--origin-column", "cohort",
            "--source", f"internal={cache_dir}",
            "--source", f"external={ext_cache_dir}",
            "--output-cache-dir", str(pooled_cache_dir),
        ]
        # os.symlink() works unprivileged on macOS/Linux (os.name == "posix")
        # but requires Administrator privileges or Developer Mode on Windows
        # ("nt"); default to a full copy there instead of failing or asking
        # the user to know this ahead of time.
        if os.name == "posix":
            merge_cmd.append("--symlink")
        if force_reprocess:
            merge_cmd.append("--overwrite")
        print(f"  {' '.join(merge_cmd)}")
        subprocess.run(merge_cmd, cwd=str(MAL_ID_LITE_ROOT), check=True)

    print(f"[5/5] Training Mal-ID-Lite ensemble on the pooled cache for "
          f"target_disease={target_disease} ...")
    run_training(
        pooled_cache_dir, pooled_dataset_name, gene_locus, target_disease, healthy_label,
        models, model2_abstention_strategy, model_save_dir, n_jobs, use_gpu, fold_ids=fold_ids,
    )

    predictions_csv = find_predictions_csv(model_save_dir)
    return convert_to_standard_scores(predictions_csv, target_disease)


def run_pipeline(
    cache_metadata, repertoire_data_dir, target_disease, dataset_name, cache_dir, model_save_dir,
    participant_col="participant_label", disease_col="disease", file_prefix="part_table_",
    file_suffix=".tsv.gz", healthy_label=HEALTHY_LABEL, gene_locus="TCR", models=(1, 2, 3),
    model2_abstention_strategy="fill_models13_mean", n_jobs=4, use_gpu=True, fold_ids=None,
    force_reprocess=False, metadata_path=None,
    ext_metadata_path=None, ext_data_dir=None, ext_cache_dir=None,
    pooled_cache_dir=None, pooled_dataset_name=None, ext_dataset_name=None,
    ext_file_template="{participant_label}_TCRB.tsv", fold_col=FOLD_COL,
):
    """Run the full consolidate -> cache -> train -> convert pipeline and
    return the standard-schema scores DataFrame.

    Plain, single-cohort case (no `ext_*` argument given): `cache_metadata`
    determines both what gets physically consolidated and what Mal-ID-Lite's
    own cache is built from -- pass the full dataset metadata (all diseases)
    when the cache is meant to be reused across multiple --target_disease
    calls sharing one cache_dir (so every participant any future target might
    need already has a file), or a single target's (optionally
    demographically-adjusted) subset when the cache is single-use (e.g. one
    cache_dir per demographic-matched run). Training itself always narrows to
    target_disease vs. healthy_label via train_ensemble.py's --diseases flag
    regardless of which was passed.

    Pooled case (`ext_metadata_path`/`ext_data_dir`/`ext_cache_dir` given --
    any one of the three is enough to trigger this path): `cache_metadata` is
    ignored entirely; `metadata_path` is used instead, since either side of
    the pool might need to build its own cache from scratch using its own
    *full* metadata (see resolve_cache). The internal cohort's own cache lives
    at `cache_dir` (unchanged meaning from the plain case); the external
    cohort's own cache lives at `ext_cache_dir`; the merged result of the two
    goes to `pooled_cache_dir`, always a separate directory from either source
    (scripts/cache_tools/merge_caches.py itself refuses to write its output
    into one of its own sources). Training happens exactly once, on the
    merged cache, using `pooled_dataset_name` to label it -- never on either
    source side standalone.
    """
    if not (MAL_ID_LITE_ROOT / "malid_lite").is_dir():
        raise FileNotFoundError(
            f"Mal-ID-Lite not found at {MAL_ID_LITE_ROOT}. Download it and "
            f"place its contents there -- see the README's Setup section "
            f"(models/Mal-ID-Lite)."
        )

    pooling = ext_metadata_path is not None or ext_data_dir is not None or ext_cache_dir is not None
    if pooling:
        return _run_pooled_pipeline(
            cache_dir=cache_dir, metadata_path=metadata_path, repertoire_data_dir=repertoire_data_dir,
            dataset_name=dataset_name,
            ext_metadata_path=ext_metadata_path, ext_data_dir=ext_data_dir, ext_cache_dir=ext_cache_dir,
            ext_dataset_name=ext_dataset_name, ext_file_template=ext_file_template,
            pooled_cache_dir=pooled_cache_dir, pooled_dataset_name=pooled_dataset_name,
            target_disease=target_disease, healthy_label=healthy_label,
            participant_col=participant_col, disease_col=disease_col, fold_col=fold_col,
            file_prefix=file_prefix, file_suffix=file_suffix, gene_locus=gene_locus, n_jobs=n_jobs,
            models=models, model2_abstention_strategy=model2_abstention_strategy,
            model_save_dir=model_save_dir, use_gpu=use_gpu, fold_ids=fold_ids,
            force_reprocess=force_reprocess,
        )

    cache_dir = Path(cache_dir)
    consolidated_dir = cache_dir / "consolidated_data"
    print(f"[1/4] Consolidating per-specimen repertoires into per-participant files "
          f"under {consolidated_dir} ...")
    has_nt_cdr3 = consolidate_participant_files(
        cache_metadata, repertoire_data_dir, consolidated_dir,
        participant_col=participant_col, file_prefix=file_prefix, file_suffix=file_suffix,
    )
    use_aa_clone_id = not has_nt_cdr3
    print(f"  Nucleotide CDR3 column present: {has_nt_cdr3} "
          f"(clone_id will use {'amino acid' if use_aa_clone_id else 'nucleotide'} sequence).")

    staged_metadata_path = cache_dir / "staged_metadata_for_malid_lite.tsv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    renamed = _rename_for_staging(cache_metadata, participant_col, disease_col)
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
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to metadata.tsv for the internal cohort. Required unless "
                             "--cache_dir already contains a built cache (a metadata.tsv "
                             "there), in which case that cache's own metadata is used "
                             "instead and this may be omitted.")
    parser.add_argument("--repertoire_data_dir", type=str, default=None,
                        help="Directory containing the internal cohort's per-specimen "
                             "repertoire .tsv.gz files. Required exactly when "
                             "--metadata_path is (both or neither).")
    parser.add_argument("--target_disease", type=str, required=True,
                        help="Disease to classify (e.g. Lupus, T1D, HIV)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset identifier (e.g. malid, rawat-t1d); used to label "
                             "and namespace the Mal-ID-Lite cache/output. For a pooled run "
                             "(any --ext_metadata_path/--ext_data_dir/--ext_cache_dir "
                             "given), this specifically names the internal cohort (e.g. "
                             "'malid') -- see --pooled_dataset_name for the pooled run's "
                             "own name, used when training on the merged result.")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Mal-ID-Lite cache directory for the internal cohort, shared "
                             "across every --target_disease run for this dataset. Built "
                             "once from --metadata_path/--repertoire_data_dir, reused "
                             "thereafter (must be the same path across those runs). For a "
                             "pooled run, this is one of the two merge sources -- never the "
                             "merged output; see --pooled_cache_dir for that.")
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
                        help="Rebuild the Mal-ID-Lite cache from scratch even if present. "
                             "For a pooled run, this only ever means 'redo the merge into "
                             "--pooled_cache_dir' -- it never rebuilds --cache_dir's or "
                             "--ext_cache_dir's own cache (those are only ever rebuilt if "
                             "no cache exists there yet at all).")
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
                        help="External cohort's metadata TSV (MAL-ID column style). "
                             "Required together with --ext_data_dir if the external "
                             "cohort's cache doesn't already exist at --ext_cache_dir; omit "
                             "both to just reuse an already-built --ext_cache_dir as-is. "
                             "Giving any of --ext_metadata_path/--ext_data_dir/"
                             "--ext_cache_dir pools a second cohort into the same "
                             "fold-based CV split as the internal cohort -- this is how the "
                             "united Zaslavsky/Mal-ID + Mitchell T1D evaluation is run. "
                             "Each side's own cache is resolved independently (reused or "
                             "built from scratch) and merged via "
                             "scripts/cache_tools/merge_caches.py; see --pooled_cache_dir "
                             "for where the merged result goes.")
    parser.add_argument("--ext_data_dir", type=str, default=None,
                        help="Directory containing the external cohort's raw repertoire "
                             "files. Required together with --ext_metadata_path (both or "
                             "neither).")
    parser.add_argument("--ext_cache_dir", type=str, default=None,
                        help="External cohort's own Mal-ID-Lite cache directory, mirroring "
                             "--cache_dir for the internal side. Required whenever pooling "
                             "is active (any --ext_metadata_path/--ext_data_dir/"
                             "--ext_cache_dir given). Built once from --ext_metadata_path/"
                             "--ext_data_dir if not already there, reused thereafter.")
    parser.add_argument("--pooled_cache_dir", type=str, default=None,
                        help="Where the merged internal+external cache goes for a pooled "
                             "run. Required whenever pooling is active. Always a distinct "
                             "path from --cache_dir/--ext_cache_dir.")
    parser.add_argument("--pooled_dataset_name", type=str, default=None,
                        help="Dataset identifier for the pooled run itself (e.g. "
                             "'malid+mitchell-t1d'), used to label training on the merged "
                             "cache. Required whenever pooling is active -- see "
                             "--dataset_name for the internal cohort's own name.")
    parser.add_argument("--ext_dataset_name", type=str, default=None,
                        help="External cohort's own name (e.g. 'mitchell-t1d'). Required "
                             "whenever pooling is active. Used only to tag each "
                             "participant's real originating cohort in a `source_dataset` "
                             "column added to the merged cache's own metadata -- the "
                             "`cohort` column utils.cohort_merge.prepare_merged_cohort() "
                             "itself sets only ever says 'internal'/'external', never the "
                             "real cohort name.")
    parser.add_argument("--ext_file_template", type=str,
                        default="{participant_label}_TCRB.tsv",
                        help="Filename template for the external cohort's raw repertoire "
                             "files, used only when building the external side from "
                             "scratch. For a cohort staged by utils.stage_disease_data, use "
                             "'part_table_{participant_label}_{specimen_label}.tsv.gz'.")
    args = parser.parse_args()

    if args.model_save_dir is None:
        parser.error("--model_save_dir is required (used as Mal-ID-Lite's training output dir)")
    if args.max_folds is not None and args.max_folds < 1:
        parser.error("--max_folds must be >= 1")

    if (args.metadata_path is None) != (args.repertoire_data_dir is None):
        parser.error("--metadata_path and --repertoire_data_dir must be given together "
                      "(both or neither).")
    internal_cache_already_built = (Path(args.cache_dir) / "metadata.tsv").exists()
    if not internal_cache_already_built and args.metadata_path is None:
        parser.error(
            f"--metadata_path and --repertoire_data_dir are required because --cache_dir "
            f"({args.cache_dir}) does not already contain a built cache."
        )

    pooling = (
        args.ext_metadata_path is not None
        or args.ext_data_dir is not None
        or args.ext_cache_dir is not None
    )
    if pooling:
        if (args.ext_metadata_path is None) != (args.ext_data_dir is None):
            parser.error("--ext_metadata_path and --ext_data_dir must be given together "
                          "(both or neither).")
        if args.ext_cache_dir is None:
            parser.error("--ext_cache_dir is required when pooling (any of "
                         "--ext_metadata_path/--ext_data_dir/--ext_cache_dir given).")
        if args.pooled_cache_dir is None:
            parser.error("--pooled_cache_dir is required when pooling.")
        if args.pooled_dataset_name is None:
            parser.error("--pooled_dataset_name is required when pooling.")
        if args.ext_dataset_name is None:
            parser.error("--ext_dataset_name is required when pooling.")
        ext_cache_already_built = (Path(args.ext_cache_dir) / "metadata.tsv").exists()
        if not ext_cache_already_built and args.ext_metadata_path is None:
            parser.error(
                f"--ext_metadata_path and --ext_data_dir are required because "
                f"--ext_cache_dir ({args.ext_cache_dir}) does not already contain a "
                f"built cache."
            )

    fold_ids = list(range(args.max_folds)) if args.max_folds is not None else None

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
        metadata_path=args.metadata_path,
        ext_metadata_path=args.ext_metadata_path,
        ext_data_dir=args.ext_data_dir,
        ext_cache_dir=args.ext_cache_dir,
        pooled_cache_dir=args.pooled_cache_dir,
        pooled_dataset_name=args.pooled_dataset_name,
        ext_dataset_name=args.ext_dataset_name,
        ext_file_template=args.ext_file_template,
        fold_col=args.fold_col,
    )

    if args.random_baseline_seeds:
        full_metadata = load_or_read_cached_metadata(
            args.metadata_path, args.cache_dir, args.fold_col, args.disease_col, args.participant_col,
        )
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
        full_metadata = load_or_read_cached_metadata(
            args.metadata_path, args.cache_dir, args.fold_col, args.disease_col, args.participant_col,
        )
        metadata_subset = prepare_metadata_subset(
            full_metadata, args.target_disease, args.disease_col, args.healthy_label,
            adjust_distribution_by_demographics=True,
        )
        scores_df = run_pipeline(
            cache_metadata=metadata_subset,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )
    elif pooling:
        # Pooling resolves each side's own cache (building it, if missing,
        # from that side's own *full* metadata) entirely inside
        # run_pipeline() itself, via metadata_path above -- not a pre-loaded/
        # pre-filtered DataFrame, unlike the other three branches here, each
        # of which already has one concrete, resolved metadata source to
        # filter before calling run_pipeline().
        scores_df = run_pipeline(
            cache_metadata=None,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )
    else:
        # No demographic adjustment, no pooling: cache_metadata is the full,
        # unfiltered dataset metadata (every disease, not just this target)
        # so that if --cache_dir is reused across multiple --target_disease
        # calls (the normal disease-classification driver pattern), every
        # participant a later target might need already has a consolidated
        # file, rather than only whichever target happened to run first.
        full_metadata = load_or_read_cached_metadata(
            args.metadata_path, args.cache_dir, args.fold_col, args.disease_col, args.participant_col,
        )
        scores_df = run_pipeline(
            cache_metadata=full_metadata,
            cache_dir=args.cache_dir, model_save_dir=args.model_save_dir,
            **pipeline_kwargs,
        )

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
