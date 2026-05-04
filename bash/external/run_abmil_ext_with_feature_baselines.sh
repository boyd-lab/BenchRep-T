#!/bin/bash
set -uo pipefail

# Run ABMIL external/cohort-merged full models and, in the same invocation,
# print/save CDR3-only and V/J-only ABMIL feature baselines.

REPO_ROOT=${REPO_ROOT:-/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench}
ABMIL_ENV=${ABMIL_ENV:-airr-bench}
DATASETS=${DATASETS:-T1D,Tb,RA}
GPUS=${GPUS:-0}
EPOCHS=${EPOCHS:-100}
MAX_PARALLEL=${MAX_PARALLEL:-}
LOGDIR=${LOGDIR:-${REPO_ROOT}/logs/abmil_ext_feature_baselines}
RESULTS=${RESULTS:-${REPO_ROOT}/results}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
MODEL_SAVE_DIR=${MODEL_SAVE_DIR:-${RESULTS}/abmil_models_ext_feature_baselines/${RUN_TS}}

mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [ ${#GPU_LIST[@]} -eq 0 ] || [ -z "${GPU_LIST[0]}" ]; then
  echo "GPUS must name at least one GPU, e.g. GPUS=0,1,2" >&2
  exit 2
fi

if [ -z "$MAX_PARALLEL" ]; then
  MAX_PARALLEL=${#GPU_LIST[@]}
fi

normalize_dataset() {
  case "$1" in
    RA|ra) echo "RA" ;;
    T1D|t1d) echo "T1D" ;;
    Tb|TB|tb) echo "Tb" ;;
    *) echo "Unknown dataset '$1'. Use T1D,Tb,RA." >&2; return 1 ;;
  esac
}

dataset_args() {
  local dataset="$1"

  case "$dataset" in
    RA)
      OUTPUT_CSV="${RESULTS}/abmil_ext_RA_full_${RUN_TS}_classification.csv"
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2"
        --target_disease "Rheumatoid Arthritis"
        --healthy_label "Healthy"
        --fold_col "CV_fold"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2"
        --ext_file_template "{participant_label}.tsv"
        --output_csv "$OUTPUT_CSV"
      )
      ;;
    T1D)
      OUTPUT_CSV="${RESULTS}/abmil_ext_T1D_full_${RUN_TS}_classification.csv"
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/malid_clean/metadata.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/malid_clean/TCR"
        --target_disease "T1D"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/T1D/metadata_T1D.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/T1D/data_processed_v2"
        --ext_file_template "{participant_label}.tsv"
        --output_csv "$OUTPUT_CSV"
      )
      ;;
    Tb)
      OUTPUT_CSV="${RESULTS}/abmil_ext_Tb_controller_progressor_full_${RUN_TS}_classification.csv"
      DATASET_ARGS=(
        --metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv"
        --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2"
        --target_disease "Progressor"
        --healthy_label "Controller"
        --fold_col "CV_fold"
        --ext_metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv"
        --ext_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2"
        --ext_file_template "{sample_name}.tsv"
        --output_csv "$OUTPUT_CSV"
      )
      ;;
  esac
}

run_dataset() {
  local dataset="$1"
  local gpu="$2"
  local log="${LOGDIR}/abmil_ext_${dataset}_full_feature_baselines_${RUN_TS}.log"
  local status

  dataset_args "$dataset"

  echo "START ${log}"
  {
    echo "run_ts=${RUN_TS}"
    echo "dataset=${dataset}"
    echo "features=full,cdr3_only,vj_only"
    echo "output_csv=${OUTPUT_CSV}"
    echo "model_save_dir=${MODEL_SAVE_DIR}"
    echo "conda_env=${ABMIL_ENV}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "epochs=${EPOCHS}"
    cd "$REPO_ROOT" || exit 1
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda run --no-capture-output -n "$ABMIL_ENV" \
      python -u -m evals.ensemble_abmil_disease_classification \
        "${DATASET_ARGS[@]}" \
        --features full \
        --include_feature_baselines \
        --epochs "$EPOCHS" \
        --model_save_dir "$MODEL_SAVE_DIR"
  } >"$log" 2>&1
  status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

active=0
status=0

for i in "${!DATASET_LIST[@]}"; do
  dataset="$(normalize_dataset "${DATASET_LIST[$i]}")" || exit 2
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"

  run_dataset "$dataset" "$gpu" &
  active=$((active + 1))

  if [ "$active" -ge "$MAX_PARALLEL" ]; then
    if ! wait -n; then
      status=1
    fi
    active=$((active - 1))
  fi
done

while [ "$active" -gt 0 ]; do
  if ! wait -n; then
    status=1
  fi
  active=$((active - 1))
done

exit "$status"
