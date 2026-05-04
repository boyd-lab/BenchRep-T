#!/bin/bash
set -uo pipefail

# Launch missing base runs:
#   - GIANA external RA and Tb in conda env "giana"
#   - Ensemble XGBoost T1D submodels in conda env "airr-bench"
#
# Stdout intentionally prints only full log paths at job start and finish.
# Model output is redirected into per-job log files.

REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/missing_base_runs
GIANA_RESULTS=${RESULTS}/giana_ext
XGB_MODEL_DIR=${RESULTS}/ensemble_xgboost_models

N_THREADS=${N_THREADS:-10}
XGB_N_JOBS=${XGB_N_JOBS:-10}
GIANA_GPUS=${GIANA_GPUS:-0,1}
XGB_GPU=${XGB_GPU:-0}
XGB_GPUS=${XGB_GPUS:-$XGB_GPU}
XGB_DEVICE=${XGB_DEVICE:-cuda}

mkdir -p "$LOGDIR" "$RESULTS" "$GIANA_RESULTS" "$XGB_MODEL_DIR"
cd "$REPO_ROOT" || exit 1

RUN_TS=$(date +%Y%m%d_%H%M%S)

IFS=',' read -r -a GIANA_GPU_LIST <<< "$GIANA_GPUS"
IFS=',' read -r -a XGB_GPU_LIST <<< "$XGB_GPUS"

run_giana_ra() {
  local gpu="$1"
  local log="${LOGDIR}/giana_base_RA_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start GIANA RA"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "use_gpu=true"
    [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
    conda run --no-capture-output -n giana python -u -m evals.giana_2021_disease_classification \
      --metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2" \
      --target_disease "Rheumatoid Arthritis" \
      --healthy_label "Healthy" \
      --fold_col "CV_fold" \
      --ext_metadata_path "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv" \
      --ext_data_dir "${REPO_ROOT}/data/external_datasets/rheumatoid_arthritis/data_processed_v2" \
      --ext_file_template "{participant_label}.tsv" \
      --output_csv "${RESULTS}/giana_2021_ext_RA_classification.csv" \
      --results_dir "$GIANA_RESULTS" \
      --n_threads "$N_THREADS" \
      --max_seqs_per_specimen 10000 \
      --exact \
      --threshold_iso 7 \
      --use_gpu
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

run_giana_tb() {
  local gpu="$1"
  local log="${LOGDIR}/giana_base_Tb_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start GIANA Tb"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "use_gpu=true"
    [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
    conda run --no-capture-output -n giana python -u -m evals.giana_2021_disease_classification \
      --metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2" \
      --target_disease "Progressor" \
      --healthy_label "Controller" \
      --fold_col "CV_fold" \
      --ext_metadata_path "${REPO_ROOT}/data/external_datasets/tuberculosis/metadata_Tb_final.tsv" \
      --ext_data_dir "${REPO_ROOT}/data/external_datasets/tuberculosis/data_processed_v2" \
      --ext_file_template "{sample_name}.tsv" \
      --output_csv "${RESULTS}/giana_2021_ext_Tb_controller_progressor_classification.csv" \
      --results_dir "$GIANA_RESULTS" \
      --n_threads "$N_THREADS" \
      --max_seqs_per_specimen 10000 \
      --exact \
      --threshold_iso 7 \
      --use_gpu
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

run_xgb_t1d() {
  local submodel="$1"
  local gpu="$2"
  local log="${LOGDIR}/ensemble_xgboost_${submodel}_T1D_${RUN_TS}.log"
  echo "START ${log}"
  {
    echo "[$(date +%T)] start Ensemble XGBoost T1D submodel=${submodel}"
    echo "log=${log}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "xgb_device=${XGB_DEVICE}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda run --no-capture-output -n airr-bench python -u -m evals.ensemble_xgboost_disease_classification \
      --metadata_path "${REPO_ROOT}/data/malid_clean/metadata.tsv" \
      --repertoire_data_dir "${REPO_ROOT}/data/malid_clean/TCR" \
      --target_disease T1D \
      --submodel "$submodel" \
      --kmer_size 4 \
      --n_jobs "$XGB_N_JOBS" \
      --xgb_device "$XGB_DEVICE" \
      --output_csv "${RESULTS}/ensemble_xgboost_${submodel}_T1D_${RUN_TS}_classification.csv" \
      --model_save_dir "$XGB_MODEL_DIR"
  } >"$log" 2>&1
  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

pids=()

run_giana_ra "${GIANA_GPU_LIST[0]:-}" &
pids+=("$!")

run_giana_tb "${GIANA_GPU_LIST[1]:-${GIANA_GPU_LIST[0]:-}}" &
pids+=("$!")

i=0
for submodel in ensemble kmer_only vj_only; do
  gpu="${XGB_GPU_LIST[$((i % ${#XGB_GPU_LIST[@]}))]}"
  run_xgb_t1d "$submodel" "$gpu" &
  pids+=("$!")
  i=$((i + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
