#!/bin/bash
set -uo pipefail

# Run DeepRC on external/cohort-merged datasets.

DEBUG=false
DATASETS="RA,T1D,Tb"
GPUS=""
PARALLEL=false
BATCH_SIZE=32
N_UPDATES=10000
EVALUATE_AT=1000
SAMPLE_N_SEQUENCES=10000

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --parallel) PARALLEL=true ;;
    --serial) PARALLEL=false ;;
    --datasets=*) DATASETS="${arg#*=}" ;;
    --gpus=*) GPUS="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --n_updates=*) N_UPDATES="${arg#*=}" ;;
    --evaluate_at=*) EVALUATE_AT="${arg#*=}" ;;
    --sample_n_sequences=*) SAMPLE_N_SEQUENCES="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if $DEBUG; then
  N_UPDATES=100
  EVALUATE_AT=50
fi

REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeprc_ext
mkdir -p "$LOGDIR" "$RESULTS"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

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
        --output_csv "${RESULTS}/deeprc_2020_ext_RA_classification.csv"
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
        --output_csv "${RESULTS}/deeprc_2020_ext_T1D_classification.csv"
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
        --output_csv "${RESULTS}/deeprc_2020_ext_Tb_controller_progressor_classification.csv"
      )
      ;;
  esac
}

run_dataset() {
  local dataset="$1"
  local gpu="${2:-}"
  local run_ts="$3"
  local log="${LOGDIR}/deeprc_ext_${dataset}_${run_ts}.log"

  dataset_args "$dataset"

  echo "[$(date +%T)] start ${dataset} DeepRC -> ${log}"
  {
    echo "[$(date +%T)] start ${dataset} DeepRC"
    device="cpu"
    if [ -n "$gpu" ]; then
      echo "Using CUDA_VISIBLE_DEVICES=${gpu}"
      export CUDA_VISIBLE_DEVICES="$gpu"
      device="cuda:0"
    fi
    cd "${REPO_ROOT}"
    python -u -m evals.deeprc_2020_disease_classification \
      "${DATASET_ARGS[@]}" \
      --results_dir "${RESULTS}/deeprc_ext" \
      --device "$device" \
      --batch_size "$BATCH_SIZE" \
      --n_updates "$N_UPDATES" \
      --evaluate_at "$EVALUATE_AT" \
      --sample_n_sequences "$SAMPLE_N_SEQUENCES"
    status=$?
    echo "[$(date +%T)] done ${dataset} DeepRC (exit ${status})"
    return $status
  } >"$log" 2>&1
}

print_planned_logs() {
  echo "DeepRC log files for run ${RUN_TS}:"
  for dataset_raw in "${DATASET_LIST[@]}"; do
    dataset="$(normalize_dataset "$dataset_raw")" || exit 2
    echo "  ${dataset}: ${LOGDIR}/deeprc_ext_${dataset}_${RUN_TS}.log"
  done
}

RUN_TS=$(date +%Y%m%d_%H%M%S)
print_planned_logs
pids=()
labels=()

for i in "${!DATASET_LIST[@]}"; do
  dataset="$(normalize_dataset "${DATASET_LIST[$i]}")" || exit 2
  gpu=""
  if [ ${#GPU_LIST[@]} -gt 0 ] && [ -n "${GPU_LIST[0]}" ]; then
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  fi

  if $PARALLEL; then
    run_dataset "$dataset" "$gpu" "$RUN_TS" &
    pids+=("$!")
    labels+=("$dataset")
  else
    run_dataset "$dataset" "$gpu" "$RUN_TS" || exit $?
  fi
done

if $PARALLEL; then
  status=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Dataset ${labels[$i]} failed; see ${LOGDIR}/deeprc_ext_${labels[$i]}_${RUN_TS}.log" >&2
      status=1
    fi
  done
  exit $status
fi
