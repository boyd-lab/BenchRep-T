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
# Internal MAL-ID data provides T1D + healthy samples; external T1D metadata
# is merged on the fly via prepare_merged_cohort to form the combined dataset.
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
MODEL_SAVE_DIR=${REPO_ROOT}/results/ensemble_xgboost_models_ext
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_ext

EXT_METADATA=${REPO_ROOT}/data/external_airr/T1D/metadata_T1D.tsv
EXT_DATA_DIR=${REPO_ROOT}/data/external_airr/T1D/data_cleaned
EXT_FILE_TEMPLATE='{participant_label}_TCRB.tsv'

TARGET_DISEASE="T1D"

mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

RUN_TS=$(date +%Y%m%d_%H%M%S)
log="${LOGDIR}/ensemble_xgboost_ext_T1D_${RUN_TS}.log"
echo "[$(date +%T)] start T1D ext classification -> $log"

{
  echo "[$(date +%T)] start T1D ext classification"

  extra_flags=()
  if $DEBUG; then
    extra_flags+=(--debug_repertoires 10)
  fi

  cd "${REPO_ROOT}"
  python -u -m evals.ensemble_xgboost_disease_classification \
    --metadata_path "$METADATA" \
    --repertoire_data_dir "$REPERTOIRE_DIR" \
    --target_disease "$TARGET_DISEASE" \
    --n_jobs "$N_JOBS" \
    --ext_metadata_path "$EXT_METADATA" \
    --ext_data_dir "$EXT_DATA_DIR" \
    --ext_file_template "$EXT_FILE_TEMPLATE" \
    --output_csv "${RESULTS}/ensemble_xgboost_ext_T1D_classification.csv" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    "${extra_flags[@]}"

  status=$?
  echo "[$(date +%T)] done T1D ext classification (exit $status)"
  exit $status
} >"$log" 2>&1

echo "[$(date +%T)] done T1D | log: $log"
