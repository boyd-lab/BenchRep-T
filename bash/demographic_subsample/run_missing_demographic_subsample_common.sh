#!/bin/bash
set -uo pipefail

# Shared launcher for missing demographic_subsample runs.
#
# Wrappers define:
#   DEFAULT_GPUS="0,1,2"
#   JOB_SPECS=("model:mode:disease" ...)
#
# Models:
#   deeprc, deeptcr, xgboost
# Modes:
#   baseline, adjust

main() {
  local debug=false
  local dry_run=false
  local debug_repertoires=10

  for arg in "$@"; do
    case "$arg" in
      --debug) debug=true ;;
      --dry-run) dry_run=true ;;
      --debug_repertoires=*) debug_repertoires="${arg#*=}" ;;
      --help|-h)
        cat <<'USAGE'
Usage: ./run_missing_demographic_subsample_<machine>.sh [--dry-run] [--debug]

Environment overrides:
  GPUS=0,1,2
  MAX_PARALLEL=<defaults to number of GPUs>
  REPO_ROOT=/path/to/airr_bench
  PYTHON_BIN=python
  CONDA_ENV=<optional, uses conda run --no-capture-output -n CONDA_ENV>
  DEEPRC_CONDA_ENV=<optional, overrides CONDA_ENV for DeepRC>
  DEEPTCR_CONDA_ENV=<optional, overrides CONDA_ENV for DeepTCR>
  XGBOOST_CONDA_ENV=<optional, overrides CONDA_ENV for xgboost>
  DEEPRC_BATCH_SIZE=32
  DEEPRC_N_UPDATES=10000
  DEEPRC_EVALUATE_AT=1000
  DEEPRC_SAMPLE_N_SEQUENCES=10000
  DEEPTCR_BATCH_SIZE=4
  XGB_N_JOBS=8
  XGB_DEVICE=cuda
USAGE
        return 0
        ;;
      *)
        echo "Unknown argument: $arg" >&2
        return 2
        ;;
    esac
  done

  local repo_root="${REPO_ROOT:-/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench}"
  local results="${RESULTS:-${repo_root}/results}"
  local logdir="${LOGDIR:-${repo_root}/logs/missing_demographic_subsample}"
  local metadata="${METADATA:-${repo_root}/data/malid_clean/metadata.tsv}"
  local repertoire_dir="${REPERTOIRE_DIR:-${repo_root}/data/malid_clean/TCR}"
  local model_save_dir="${MODEL_SAVE_DIR:-${results}/ensemble_xgboost_models_demographic}"
  local run_ts="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
  local gpustr="${GPUS:-${DEFAULT_GPUS:-0}}"
  local python_bin="${PYTHON_BIN:-python}"
  local max_parallel="${MAX_PARALLEL:-}"

  local deeprc_batch_size="${DEEPRC_BATCH_SIZE:-32}"
  local deeprc_n_updates="${DEEPRC_N_UPDATES:-10000}"
  local deeprc_evaluate_at="${DEEPRC_EVALUATE_AT:-1000}"
  local deeprc_sample_n_sequences="${DEEPRC_SAMPLE_N_SEQUENCES:-10000}"
  local deeptcr_batch_size="${DEEPTCR_BATCH_SIZE:-4}"
  local xgb_n_jobs="${XGB_N_JOBS:-8}"
  local xgb_device="${XGB_DEVICE:-cuda}"

  IFS=',' read -r -a gpu_list <<< "$gpustr"
  if [ ${#gpu_list[@]} -eq 0 ] || [ -z "${gpu_list[0]}" ]; then
    echo "GPUS must name at least one GPU, e.g. GPUS=0,1,2" >&2
    return 2
  fi
  if [ -z "$max_parallel" ]; then
    max_parallel=${#gpu_list[@]}
  fi
  if [ "${#JOB_SPECS[@]}" -eq 0 ]; then
    echo "JOB_SPECS is empty" >&2
    return 2
  fi

  if $debug; then
    deeprc_n_updates=100
    deeprc_evaluate_at=50
  fi

  mkdir -p "$logdir" "$results" "$model_save_dir" \
    "${results}/deeprc_demographic" "${results}/deeptcr_demographic"
  cd "$repo_root" || return 1

  echo "run_ts=${run_ts}"
  echo "logdir=${logdir}"
  echo "gpus=${gpustr}"
  echo "max_parallel=${max_parallel}"
  echo "jobs=${#JOB_SPECS[@]}"

  local active=0
  local status=0
  local i

  for i in "${!JOB_SPECS[@]}"; do
    local spec="${JOB_SPECS[$i]}"
    local gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"

    run_job "$spec" "$gpu" "$run_ts" "$repo_root" "$results" "$logdir" \
      "$metadata" "$repertoire_dir" "$model_save_dir" "$python_bin" \
      "$deeprc_batch_size" "$deeprc_n_updates" "$deeprc_evaluate_at" \
      "$deeprc_sample_n_sequences" "$deeptcr_batch_size" \
      "$xgb_n_jobs" "$xgb_device" "$debug" "$debug_repertoires" "$dry_run" &
    active=$((active + 1))

    if [ "$active" -ge "$max_parallel" ]; then
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

  return "$status"
}

run_job() {
  local spec="$1"
  local gpu="$2"
  local run_ts="$3"
  local repo_root="$4"
  local results="$5"
  local logdir="$6"
  local metadata="$7"
  local repertoire_dir="$8"
  local model_save_dir="$9"
  local python_bin="${10}"
  local deeprc_batch_size="${11}"
  local deeprc_n_updates="${12}"
  local deeprc_evaluate_at="${13}"
  local deeprc_sample_n_sequences="${14}"
  local deeptcr_batch_size="${15}"
  local xgb_n_jobs="${16}"
  local xgb_device="${17}"
  local debug="${18}"
  local debug_repertoires="${19}"
  local dry_run="${20}"

  local model mode disease
  IFS=':' read -r model mode disease <<< "$spec"

  if [ -z "$model" ] || [ -z "$mode" ] || [ -z "$disease" ]; then
    echo "Invalid job spec: $spec" >&2
    return 2
  fi

  local log="${logdir}/${model}_demographic_${mode}_${disease}_${run_ts}.log"
  echo "START ${log} gpu=${gpu} spec=${spec}"

  if $dry_run; then
    print_command "$model" "$mode" "$disease" "$gpu" "$run_ts" "$repo_root" \
      "$results" "$metadata" "$repertoire_dir" "$model_save_dir" "$python_bin" \
      "$deeprc_batch_size" "$deeprc_n_updates" "$deeprc_evaluate_at" \
      "$deeprc_sample_n_sequences" "$deeptcr_batch_size" "$xgb_n_jobs" \
      "$xgb_device" "$debug" "$debug_repertoires"
    echo "DONE ${log} exit=0"
    return 0
  fi

  {
    echo "run_ts=${run_ts}"
    echo "spec=${spec}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "log=${log}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_model "$model" "$mode" "$disease" "$repo_root" "$results" "$metadata" \
      "$repertoire_dir" "$model_save_dir" "$python_bin" "$deeprc_batch_size" \
      "$deeprc_n_updates" "$deeprc_evaluate_at" "$deeprc_sample_n_sequences" \
      "$deeptcr_batch_size" "$xgb_n_jobs" "$xgb_device" "$debug" \
      "$debug_repertoires"
  } >"$log" 2>&1

  local status=$?
  echo "DONE ${log} exit=${status}"
  return "$status"
}

python_cmd() {
  local model="$1"
  local python_bin="$2"
  local env_var=""

  case "$model" in
    deeprc) env_var="${DEEPRC_CONDA_ENV:-${CONDA_ENV:-}}" ;;
    deeptcr) env_var="${DEEPTCR_CONDA_ENV:-${CONDA_ENV:-}}" ;;
    xgboost) env_var="${XGBOOST_CONDA_ENV:-${CONDA_ENV:-}}" ;;
    *) echo "Unknown model: $model" >&2; return 2 ;;
  esac

  if [ -n "$env_var" ]; then
    PY_CMD=(conda run --no-capture-output -n "$env_var" "$python_bin")
  else
    PY_CMD=("$python_bin")
  fi
}

mode_flags() {
  local mode="$1"
  case "$mode" in
    baseline) MODE_FLAGS=(--random_baseline_seeds 7 14 21 28 35) ;;
    adjust) MODE_FLAGS=(--adjust_distribution_by_demographics) ;;
    *) echo "Unknown mode: $mode" >&2; return 2 ;;
  esac
}

run_model() {
  local model="$1"
  local mode="$2"
  local disease="$3"
  local repo_root="$4"
  local results="$5"
  local metadata="$6"
  local repertoire_dir="$7"
  local model_save_dir="$8"
  local python_bin="$9"
  local deeprc_batch_size="${10}"
  local deeprc_n_updates="${11}"
  local deeprc_evaluate_at="${12}"
  local deeprc_sample_n_sequences="${13}"
  local deeptcr_batch_size="${14}"
  local xgb_n_jobs="${15}"
  local xgb_device="${16}"
  local debug="${17}"
  local debug_repertoires="${18}"

  python_cmd "$model" "$python_bin" || return $?
  mode_flags "$mode" || return $?

  echo "command=${PY_CMD[*]} -m evals.${model} ..."

  case "$model" in
    deeprc)
      "${PY_CMD[@]}" -u -m evals.deeprc_2020_disease_classification \
        --metadata_path "$metadata" \
        --repertoire_data_dir "$repertoire_dir" \
        --target_disease "$disease" \
        --output_csv "${results}/deeprc_2020_${mode}_${disease}_${run_ts}_classification.csv" \
        --results_dir "${results}/deeprc_demographic" \
        --device "cuda:0" \
        --batch_size "$deeprc_batch_size" \
        --n_updates "$deeprc_n_updates" \
        --evaluate_at "$deeprc_evaluate_at" \
        --sample_n_sequences "$deeprc_sample_n_sequences" \
        "${MODE_FLAGS[@]}"
      ;;
    deeptcr)
      local debug_flags=()
      if $debug; then
        debug_flags=(--debug --debug_repertoires "$debug_repertoires" --epochs_max 5)
      fi
      "${PY_CMD[@]}" -u -m evals.deeptcr_2021_disease_classification \
        --metadata_path "$metadata" \
        --repertoire_data_dir "$repertoire_dir" \
        --target_disease "$disease" \
        --output_csv "${results}/deeptcr_${mode}_${disease}_${run_ts}_classification.csv" \
        --results_dir "${results}/deeptcr_demographic" \
        --batch_size "$deeptcr_batch_size" \
        --device 0 \
        "${debug_flags[@]}" \
        "${MODE_FLAGS[@]}"
      ;;
    xgboost)
      local debug_flags=()
      if $debug; then
        debug_flags=(--debug_repertoires "$debug_repertoires")
      fi
      "${PY_CMD[@]}" -u -m evals.ensemble_xgboost_disease_classification \
        --metadata_path "$metadata" \
        --repertoire_data_dir "$repertoire_dir" \
        --target_disease "$disease" \
        --n_jobs "$xgb_n_jobs" \
        --xgb_device "$xgb_device" \
        --output_csv "${results}/ensemble_xgboost_${mode}_${disease}_${run_ts}_classification.csv" \
        --model_save_dir "$model_save_dir" \
        "${debug_flags[@]}" \
        "${MODE_FLAGS[@]}"
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 2
      ;;
  esac
}

print_command() {
  local model="$1"
  local mode="$2"
  local disease="$3"
  local gpu="$4"
  local run_ts="$5"
  local repo_root="$6"
  local results="$7"
  local metadata="$8"
  local repertoire_dir="$9"
  local model_save_dir="${10}"
  local python_bin="${11}"
  local deeprc_batch_size="${12}"
  local deeprc_n_updates="${13}"
  local deeprc_evaluate_at="${14}"
  local deeprc_sample_n_sequences="${15}"
  local deeptcr_batch_size="${16}"
  local xgb_n_jobs="${17}"
  local xgb_device="${18}"
  local debug="${19}"
  local debug_repertoires="${20}"

  mode_flags "$mode" || return $?
  python_cmd "$model" "$python_bin" || return $?

  echo "DRY_RUN spec=${model}:${mode}:${disease}"
  echo "DRY_RUN CUDA_VISIBLE_DEVICES=${gpu}"
  case "$model" in
    deeprc)
      echo "DRY_RUN ${PY_CMD[*]} -u -m evals.deeprc_2020_disease_classification --metadata_path ${metadata} --repertoire_data_dir ${repertoire_dir} --target_disease ${disease} --output_csv ${results}/deeprc_2020_${mode}_${disease}_${run_ts}_classification.csv --results_dir ${results}/deeprc_demographic --device cuda:0 --batch_size ${deeprc_batch_size} --n_updates ${deeprc_n_updates} --evaluate_at ${deeprc_evaluate_at} --sample_n_sequences ${deeprc_sample_n_sequences} ${MODE_FLAGS[*]}"
      ;;
    deeptcr)
      local debug_flags=()
      if $debug; then
        debug_flags=(--debug --debug_repertoires "$debug_repertoires" --epochs_max 5)
      fi
      echo "DRY_RUN ${PY_CMD[*]} -u -m evals.deeptcr_2021_disease_classification --metadata_path ${metadata} --repertoire_data_dir ${repertoire_dir} --target_disease ${disease} --output_csv ${results}/deeptcr_${mode}_${disease}_${run_ts}_classification.csv --results_dir ${results}/deeptcr_demographic --batch_size ${deeptcr_batch_size} --device 0 ${debug_flags[*]} ${MODE_FLAGS[*]}"
      ;;
    xgboost)
      local debug_flags=()
      if $debug; then
        debug_flags=(--debug_repertoires "$debug_repertoires")
      fi
      echo "DRY_RUN ${PY_CMD[*]} -u -m evals.ensemble_xgboost_disease_classification --metadata_path ${metadata} --repertoire_data_dir ${repertoire_dir} --target_disease ${disease} --n_jobs ${xgb_n_jobs} --xgb_device ${xgb_device} --output_csv ${results}/ensemble_xgboost_${mode}_${disease}_${run_ts}_classification.csv --model_save_dir ${model_save_dir} ${debug_flags[*]} ${MODE_FLAGS[*]}"
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 2
      ;;
  esac
}
