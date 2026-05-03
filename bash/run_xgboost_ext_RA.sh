#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
N_JOBS=10

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --n_jobs=*) N_JOBS="${arg#*=}" ;;
  esac
done

# ---- config ----
# Only RA external data is used; no MAL-ID samples are included.
# Existing metadata_RA_final.tsv already has disease labels and folds.
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
MODEL_SAVE_DIR=${REPO_ROOT}/results/ensemble_xgboost_models_ext
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_ext

EXT_METADATA=${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv
EXT_DATA_DIR=${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_cleaned
EXT_FILE_TEMPLATE='{sample_name}.tsv'

TARGET_DISEASE="Rheumatoid Arthritis"
HEALTHY_LABEL="Healthy"
FOLD_COL="CV_fold"

mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

RUN_TS=$(date +%Y%m%d_%H%M%S)
log="${LOGDIR}/ensemble_xgboost_ext_RA_${RUN_TS}.log"
echo "[$(date +%T)] start RA ext classification -> $log"

{
  echo "[$(date +%T)] start RA ext classification"

  extra_flags=()
  if $DEBUG; then
    extra_flags+=(--debug_repertoires 10)
  fi

  cd "${REPO_ROOT}"
  python -u -m evals.ensemble_xgboost_disease_classification \
    --metadata_path "$EXT_METADATA" \
    --repertoire_data_dir "$EXT_DATA_DIR" \
    --target_disease "$TARGET_DISEASE" \
    --healthy_label "$HEALTHY_LABEL" \
    --fold_col "$FOLD_COL" \
    --n_jobs "$N_JOBS" \
    --ext_metadata_path "$EXT_METADATA" \
    --ext_data_dir "$EXT_DATA_DIR" \
    --ext_file_template "$EXT_FILE_TEMPLATE" \
    --output_csv "${RESULTS}/ensemble_xgboost_ext_RA_classification.csv" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    "${extra_flags[@]}"

  status=$?
  echo "[$(date +%T)] done RA ext classification (exit $status)"
  exit $status
} >"$log" 2>&1

echo "[$(date +%T)] done RA | log: $log"
