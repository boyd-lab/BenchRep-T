#!/bin/bash
set -uo pipefail

# Run GIANA on external/cohort-merged datasets.

DEBUG=false
DEBUG_REPERTOIRES=10
DATASETS="RA,T1D,Tb"
PARALLEL=false
N_THREADS=10
USE_GPU=false

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --parallel) PARALLEL=true ;;
    --serial) PARALLEL=false ;;
    --datasets=*) DATASETS="${arg#*=}" ;;
    --n_threads=*) N_THREADS="${arg#*=}" ;;
    --cpu) USE_GPU=false ;;
    --gpu) USE_GPU=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana_ext
mkdir -p "$LOGDIR" "$RESULTS/giana_ext"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

normalize_dataset() {
  case "$1" in
    RA|ra) echo "RA" ;;
    T1D|t1d) echo "T1D" ;;
    Tb|TB|tb) echo "Tb" ;;
    *) echo "Unknown dataset '$1'. Use RA,T1D,Tb." >&2; return 1 ;;
  esac
}

dataset_args() {
  case "$1" in
    RA)
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2"
        --target_disease "Rheumatoid Arthritis"
        --healthy_label "Healthy"
        --fold_col "CV_fold"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2"
        --ext_file_template "{participant_label}.tsv"
        --output_csv "${RESULTS}/giana_2021_ext_RA_classification.csv"
      )
      ;;
    T1D)
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/malid_clean/metadata.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/malid_clean/TCR"
        --target_disease "T1D"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/T1D/metadata_T1D.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/T1D/data_processed_v2"
        --ext_file_template "{participant_label}.tsv"
        --output_csv "${RESULTS}/giana_2021_ext_T1D_classification.csv"
      )
      ;;
    Tb)
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2"
        --target_disease "Progressor"
        --healthy_label "Controller"
        --fold_col "CV_fold"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2"
        --ext_file_template "{sample_name}.tsv"
        --output_csv "${RESULTS}/giana_2021_ext_Tb_controller_progressor_classification.csv"
      )
      ;;
  esac
}

run_dataset() {
  local dataset="$1"
  local run_ts="$2"
  local log="${LOGDIR}/giana_ext_${dataset}_${run_ts}.log"

  dataset_args "$dataset"

  extra_flags=()
  $USE_GPU && extra_flags+=(--use_gpu)
  if $DEBUG; then
    extra_flags+=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")
  fi

  echo "[$(date +%T)] start ${dataset} GIANA -> ${log}"
  {
    echo "[$(date +%T)] start ${dataset} GIANA"
    cd "${REPO_ROOT}"
    python -u -m evals.giana_2021_disease_classification \
      "${DATASET_ARGS[@]}" \
      --results_dir "${RESULTS}/giana_ext" \
      --n_threads "$N_THREADS" \
      --max_seqs_per_specimen 10000 \
      --exact \
      --threshold_iso 7 \
      "${extra_flags[@]}"
    status=$?
    echo "[$(date +%T)] done ${dataset} GIANA (exit ${status})"
    return $status
  } >"$log" 2>&1
}

RUN_TS=$(date +%Y%m%d_%H%M%S)
pids=()
labels=()

for dataset_raw in "${DATASET_LIST[@]}"; do
  dataset="$(normalize_dataset "$dataset_raw")" || exit 2
  if $PARALLEL; then
    run_dataset "$dataset" "$RUN_TS" &
    pids+=("$!")
    labels+=("$dataset")
  else
    run_dataset "$dataset" "$RUN_TS" || exit $?
  fi
done

if $PARALLEL; then
  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Dataset ${labels[$i]} failed; see ${LOGDIR}/giana_ext_${labels[$i]}_${RUN_TS}.log" >&2
      status=1
    fi
  done
  exit $status
fi
