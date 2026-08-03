"""
Evaluation script for Mal-ID-Lite disease classification.

Wraps the vendored Mal-ID-Lite submodule (models/Mal-ID-Lite), a streamlined
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
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MAL_ID_LITE_ROOT = Path(__file__).resolve().parent.parent / "models" / "Mal-ID-Lite"
HEALTHY_LABEL = "Healthy/Background"


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path, sep="\t")


def consolidate_participant_files(
    metadata,
    repertoire_data_dir,
    data_dir_out,
    participant_col="participant_label",
    file_prefix="part_table_",
    file_suffix=".tsv.gz",
):
    """Build one part_table_{participant}{file_suffix} per participant under
    data_dir_out, concatenating that participant's specimen files and setting
    `repertoire_id` = specimen_label on every row. Idempotent: participants
    whose consolidated file already exists are skipped entirely (this is what
    makes repeated calls -- once per target disease in the same dataset --
    cheap after the first).

    Returns True if a genuine nucleotide `cdr3` column was found in the
    source data, so the caller can decide whether Mal-ID-Lite's default
    clone_id computation (nucleotide-based) will work, or whether
    --clone-id-use-aa is needed.
    """
    data_dir_out = Path(data_dir_out)
    data_dir_out.mkdir(parents=True, exist_ok=True)
    is_gz = file_suffix.endswith(".gz")
    has_nt_cdr3 = None
    an_existing_out_path = None
    built, skipped, missing = 0, 0, 0

    for participant, group in metadata.groupby(participant_col):
        out_path = data_dir_out / f"{file_prefix}{participant}{file_suffix}"
        if out_path.exists():
            skipped += 1
            an_existing_out_path = out_path
            continue
        frames = []
        for _, row in group.iterrows():
            specimen = row["specimen_label"]
            src = Path(repertoire_data_dir) / f"{file_prefix}{participant}_{specimen}{file_suffix}"
            if not src.exists():
                missing += 1
                continue
            df = pd.read_csv(
                src, sep="\t", compression="gzip" if is_gz else None, low_memory=False
            )
            if has_nt_cdr3 is None:
                has_nt_cdr3 = "cdr3" in df.columns
            df["repertoire_id"] = specimen
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
        # source file was read this call. Check an existing consolidated file
        # instead of silently defaulting to False (which would wrongly force
        # the AA clone_id fallback on every run after the first).
        header = pd.read_csv(
            an_existing_out_path, sep="\t",
            compression="gzip" if is_gz else None, nrows=0,
        )
        has_nt_cdr3 = "cdr3" in header.columns

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
    models, model2_abstention_strategy, output_dir, n_jobs, use_gpu,
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
    parser.add_argument("--fold_col", type=str,
                        default="malid_cross_validation_fold_id_when_in_test_set")
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
    args = parser.parse_args()

    if args.model_save_dir is None:
        parser.error("--model_save_dir is required (used as Mal-ID-Lite's training output dir)")

    metadata = load_metadata(args.metadata_path)
    metadata = metadata[
        metadata[args.disease_col].isin([args.target_disease, args.healthy_label])
    ].copy()

    consolidated_dir = Path(args.cache_dir) / "consolidated_data"
    print(f"[1/4] Consolidating per-specimen repertoires into per-participant files "
          f"under {consolidated_dir} ...")
    has_nt_cdr3 = consolidate_participant_files(
        metadata, args.repertoire_data_dir, consolidated_dir,
        participant_col=args.participant_col,
        file_prefix=args.file_prefix, file_suffix=args.file_suffix,
    )
    use_aa_clone_id = not has_nt_cdr3
    print(f"  Nucleotide CDR3 column present: {has_nt_cdr3} "
          f"(clone_id will use {'amino acid' if use_aa_clone_id else 'nucleotide'} sequence).")

    staged_metadata_path = Path(args.cache_dir) / "staged_metadata_for_malid_lite.tsv"
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    full_metadata = load_metadata(args.metadata_path)
    full_metadata.rename(columns={args.disease_col: "disease"}, inplace=True) \
        if args.disease_col != "disease" else None
    full_metadata.to_csv(staged_metadata_path, sep="\t", index=False)

    print(f"[2/4] Building Mal-ID-Lite cache at {args.cache_dir} (skipped if already built) ...")
    build_cache_if_missing(
        consolidated_dir, staged_metadata_path, args.cache_dir, args.gene_locus,
        args.n_jobs, use_aa_clone_id, args.force_reprocess,
    )

    print(f"[3/4] Training Mal-ID-Lite ensemble for target_disease={args.target_disease} ...")
    run_training(
        args.cache_dir, args.dataset_name, args.gene_locus, args.target_disease,
        args.healthy_label, args.models, args.model2_abstention_strategy,
        args.model_save_dir, args.n_jobs, not args.no_gpu,
    )

    print("[4/4] Converting ensemble_predictions.csv to standard scores.csv schema ...")
    predictions_csv = find_predictions_csv(args.model_save_dir)
    scores_df = convert_to_standard_scores(predictions_csv, args.target_disease)

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"\nScores saved to: {args.output_csv}")
