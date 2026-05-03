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
# Only Tb external data is used; no MAL-ID samples are included.
# Controller is the negative class; Progressor is the positive class.
# Existing metadata_Tb_final.tsv already has labels and folds.
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
MODEL_SAVE_DIR=${REPO_ROOT}/results/ensemble_xgboost_models_ext
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_ext

EXT_METADATA=${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv
EXT_DATA_DIR=${REPO_ROOT}/data/external_datasets/tuberculosis/data_cleaned
EXT_FILE_TEMPLATE='{sample_name}.tsv'

TARGET_DISEASE="Progressor"
HEALTHY_LABEL="Controller"
FOLD_COL="CV_fold"

mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

RUN_TS=$(date +%Y%m%d_%H%M%S)
log="${LOGDIR}/ensemble_xgboost_ext_Tb_${RUN_TS}.log"
echo "[$(date +%T)] start Tb ext classification -> $log"

{
  echo "[$(date +%T)] start Tb ext classification"

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
    --output_csv "${RESULTS}/ensemble_xgboost_ext_Tb_controller_progressor_classification.csv" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    "${extra_flags[@]}"

  status=$?
  echo "[$(date +%T)] done Tb ext classification (exit $status)"
  exit $status
} >"$log" 2>&1

echo "[$(date +%T)] done Tb | log: $log"
