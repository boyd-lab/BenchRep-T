#!/bin/bash
set -uo pipefail

# Launch missing merged/base runs:
#   - DeepRC T1D merged
#   - DeepTCR T1D merged
#   - DeepTCR Tb external/controller-progressor
#
# Stdout intentionally prints only full log paths at job start and finish.
# Model output is redirected into per-job log files.

REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/missing_deeprc_deeptcr_runs
DEEPRC_RESULTS=${RESULTS}/deeprc_ext
DEEPTCR_RESULTS=${RESULTS}/deeptcr_ext

DEEPRC_CONDA_ENV=${DEEPRC_CONDA_ENV:-deeprc}
DEEPTCR_CONDA_ENV=${DEEPTCR_CONDA_ENV:-deeptcr}
GPUS=${GPUS:-0,1,3}

DEEPRC_BATCH_SIZE=${DEEPRC_BATCH_SIZE:-32}
DEEPRC_N_UPDATES=${DEEPRC_N_UPDATES:-10000}
DEEPRC_EVALUATE_AT=${DEEPRC_EVALUATE_AT:-1000}
DEEPRC_SAMPLE_N_SEQUENCES=${DEEPRC_SAMPLE_N_SEQUENCES:-10000}

DEEPTCR_BATCH_SIZE=${DEEPTCR_BATCH_SIZE:-4}

mkdir -p "$LOGDIR" "$RESULTS" "$DEEPRC_RESULTS" "$DEEPTCR_RESULTS"
cd "$REPO_ROOT" || exit 1

RUN_TS=$(date +%Y%m%d_%H%M%S)
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

run_deeprc_t1d() {
  local gpu="$1"
  local log="${LOGDIR}/deeprc_merged_T1D_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start DeepRC merged T1D"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "conda_env=${DEEPRC_CONDA_ENV}"
    conda run --no-capture-output -n "$DEEPRC_CONDA_ENV" python -u -m evals.deeprc_2020_disease_classification \
      --metadata_path "${REPO_ROOT}/data/malid_clean/metadata.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/malid_clean/TCR" \
      --target_disease "T1D" \
      --ext_metadata_path "${REPO_ROOT}/data/external_datasets/T1D/metadata_T1D.tsv" \
      --ext_data_dir "${REPO_ROOT}/data/external_datasets/T1D/data_processed_v2" \
      --ext_file_template "{participant_label}.tsv" \
      --output_csv "${RESULTS}/deeprc_2020_ext_T1D_classification.csv" \
      --results_dir "$DEEPRC_RESULTS" \
      --device "cuda:0" \
      --batch_size "$DEEPRC_BATCH_SIZE" \
      --n_updates "$DEEPRC_N_UPDATES" \
      --evaluate_at "$DEEPRC_EVALUATE_AT" \
      --sample_n_sequences "$DEEPRC_SAMPLE_N_SEQUENCES"
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

run_deeptcr_t1d() {
  local gpu="$1"
  local log="${LOGDIR}/deeptcr_merged_T1D_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start DeepTCR merged T1D"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "conda_env=${DEEPTCR_CONDA_ENV}"
    conda run --no-capture-output -n "$DEEPTCR_CONDA_ENV" python -u -m evals.deeptcr_2021_disease_classification \
      --metadata_path "${REPO_ROOT}/data/malid_clean/metadata.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/malid_clean/TCR" \
      --target_disease "T1D" \
      --ext_metadata_path "${REPO_ROOT}/data/external_datasets/T1D/metadata_T1D.tsv" \
      --ext_data_dir "${REPO_ROOT}/data/external_datasets/T1D/data_processed_v2" \
      --ext_file_template "{participant_label}.tsv" \
      --output_csv "${RESULTS}/deeptcr_2021_ext_T1D_classification.csv" \
      --results_dir "$DEEPTCR_RESULTS" \
      --batch_size "$DEEPTCR_BATCH_SIZE" \
      --device 0
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

run_deeptcr_tb() {
  local gpu="$1"
  local log="${LOGDIR}/deeptcr_Tb_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start DeepTCR Tb"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "conda_env=${DEEPTCR_CONDA_ENV}"
    conda run --no-capture-output -n "$DEEPTCR_CONDA_ENV" python -u -m evals.deeptcr_2021_disease_classification \
      --metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2" \
      --target_disease "Progressor" \
      --healthy_label "Controller" \
      --fold_col "CV_fold" \
      --ext_metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv" \
      --ext_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2" \
      --ext_file_template "{sample_name}.tsv" \
      --output_csv "${RESULTS}/deeptcr_2021_ext_Tb_controller_progressor_classification.csv" \
      --results_dir "$DEEPTCR_RESULTS" \
      --batch_size "$DEEPTCR_BATCH_SIZE" \
      --device 0
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

pids=()

run_deeprc_t1d "${GPU_LIST[0]:-0}" &
pids+=("$!")

run_deeptcr_t1d "${GPU_LIST[1]:-1}" &
pids+=("$!")

run_deeptcr_tb "${GPU_LIST[2]:-3}" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
