#!/bin/bash
set -uo pipefail

# CPU-only runner for remaining Ensemble XGBoost experiments.
#
# Experiments:
#   1. Demographic subsample:
#      - adjust:   Lupus, HIV, Influenza, Covid19
#      - baseline: Lupus, HIV, Influenza, Covid19, seeds 7 14 21 28 35
#   2. Sequencing-depth scaling:
#      - Lupus, HIV using data/depth_indices/depth_indices_max75k.json.gz
#        depths: 1000, 5000, 10000, 25000, 50000, 75000; 5 repeats
#   3. Driver sequence:
#      - retrain base Ensemble XGBoost models for Influenza, Covid19
#      - run driver identification at k=100,1000,10000

RUN_DEMOGRAPHIC=true
RUN_SCALING=true
RUN_DRIVER=true
DRY_RUN=false
SMOKE_TEST=false

DEMO_PARALLEL=4
DEMO_N_JOBS=20
SCALING_PARALLEL=2
SCALING_N_JOBS=30
DRIVER_TRAIN_PARALLEL=2
DRIVER_TRAIN_N_JOBS=30
DRIVER_SCORE_PARALLEL=2
DEBUG_REPERTOIRES=0
BASELINE_SEEDS=(7 14 21 28 35)
DRIVER_K="100,1000,10000"
DRIVER_K_TAG="100_1000_10000"
DRIVER_MAX_REPERTOIRES=()

for arg in "$@"; do
  case "$arg" in
    --phase=demographic)
      RUN_DEMOGRAPHIC=true
      RUN_SCALING=false
      RUN_DRIVER=false
      ;;
    --phase=scaling)
      RUN_DEMOGRAPHIC=false
      RUN_SCALING=true
      RUN_DRIVER=false
      ;;
    --phase=driver)
      RUN_DEMOGRAPHIC=false
      RUN_SCALING=false
      RUN_DRIVER=true
      ;;
    --phase=all)
      RUN_DEMOGRAPHIC=true
      RUN_SCALING=true
      RUN_DRIVER=true
      ;;
    --dry_run)
      DRY_RUN=true
      ;;
    --smoke_test)
      SMOKE_TEST=true
      ;;
    --demo_parallel=*) DEMO_PARALLEL="${arg#*=}" ;;
    --demo_n_jobs=*) DEMO_N_JOBS="${arg#*=}" ;;
    --scaling_parallel=*) SCALING_PARALLEL="${arg#*=}" ;;
    --scaling_n_jobs=*) SCALING_N_JOBS="${arg#*=}" ;;
    --driver_train_parallel=*) DRIVER_TRAIN_PARALLEL="${arg#*=}" ;;
    --driver_train_n_jobs=*) DRIVER_TRAIN_N_JOBS="${arg#*=}" ;;
    --driver_score_parallel=*) DRIVER_SCORE_PARALLEL="${arg#*=}" ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if $SMOKE_TEST; then
  DEMO_PARALLEL=1
  DEMO_N_JOBS=1
  SCALING_PARALLEL=1
  SCALING_N_JOBS=1
  DRIVER_TRAIN_PARALLEL=1
  DRIVER_TRAIN_N_JOBS=1
  DRIVER_SCORE_PARALLEL=1
  DEBUG_REPERTOIRES="${DEBUG_REPERTOIRES:-6}"
  if [[ "$DEBUG_REPERTOIRES" -eq 0 ]]; then
    DEBUG_REPERTOIRES=6
  fi
  BASELINE_SEEDS=(7)
  DRIVER_K="1,5"
  DRIVER_K_TAG="1_5"
  DRIVER_MAX_REPERTOIRES=(--max_repertoires 12)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DEPTH_INDICES=${REPO_ROOT}/data/depth_indices/depth_indices_max75k.json.gz
DRIVER_SEQS=${REPO_ROOT}/data/public_clones/vdjdb_matches_expanded.csv
RESULTS=${REPO_ROOT}/results
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_TS=$(date +%Y%m%d_%H%M%S)

DEMO_LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_demographic_cpu
SCALING_LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_scaling_cpu
DRIVER_LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_driver_cpu
DEMO_MODEL_DIR=${RESULTS}/ensemble_xgboost_models_demographic_cpu_${RUN_TS}
DRIVER_MODEL_DIR=${RESULTS}/ensemble_xgboost_models_driver_cpu_${RUN_TS}

mkdir -p "$RESULTS" "$DEMO_LOGDIR" "$SCALING_LOGDIR" "$DRIVER_LOGDIR"
mkdir -p "$DEMO_MODEL_DIR" "$DRIVER_MODEL_DIR"

cd "$REPO_ROOT" || exit 1

run_cmd() {
  local log="$1"
  shift

  echo "[$(date +%T)] log: $log"
  printf '[%s] command:' "$(date +%T)" >"$log"
  printf ' %q' "$@" >>"$log"
  printf '\n' >>"$log"

  if $DRY_RUN; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  "$@" >>"$log" 2>&1
}

run_pool() {
  local max_parallel="$1"
  local job_func="$2"
  shift 2
  local specs=("$@")
  local fifo
  local status=0

  fifo=$(mktemp -u)
  mkfifo "$fifo"
  exec 3<>"$fifo"
  rm -f "$fifo"

  for i in $(seq 1 "$max_parallel"); do
    echo "$i" >&3
  done

  for spec in "${specs[@]}"; do
    read -r slot <&3
    {
      "$job_func" "$spec" "$slot"
      job_status=$?
      echo "$slot" >&3
      exit "$job_status"
    } &
  done

  for pid in $(jobs -p); do
    wait "$pid" || status=1
  done

  exec 3>&-
  exec 3<&-
  return "$status"
}

run_demographic_one() {
  local spec="$1"
  local slot="$2"
  local mode="${spec%%:*}"
  local disease="${spec#*:}"
  local log="${DEMO_LOGDIR}/ensemble_xgboost_demographic_${mode}_${disease}_${RUN_TS}.log"
  local output="${RESULTS}/ensemble_xgboost_${mode}_${disease}_${RUN_TS}_classification.csv"
  local args=()

  if [[ "$mode" == "baseline" ]]; then
    args+=(--random_baseline_seeds "${BASELINE_SEEDS[@]}")
  else
    args+=(--adjust_distribution_by_demographics)
  fi
  if [[ "$DEBUG_REPERTOIRES" -gt 0 ]]; then
    args+=(--debug_repertoires "$DEBUG_REPERTOIRES")
  fi

  echo "[$(date +%T)] start demographic $mode $disease slot=$slot"
  run_cmd "$log" \
    "$PYTHON_BIN" -u -m evals.ensemble_xgboost_disease_classification \
      --metadata_path "$METADATA" \
      --repertoire_data_dir "$REPERTOIRE_DIR" \
      --target_disease "$disease" \
      --n_jobs "$DEMO_N_JOBS" \
      --xgb_device cpu \
      --output_csv "$output" \
      --model_save_dir "$DEMO_MODEL_DIR" \
      "${args[@]}"
  local status=$?
  echo "[$(date +%T)] done demographic $mode $disease exit=$status log=$log"
  return "$status"
}

run_scaling_one() {
  local disease="$1"
  local slot="$2"
  local log="${SCALING_LOGDIR}/ensemble_xgboost_scaling_${disease}_${RUN_TS}.log"
  local output="${RESULTS}/ensemble_xgboost_${disease}_scaling_${RUN_TS}.json"

  echo "[$(date +%T)] start scaling $disease slot=$slot"
  local debug_args=()
  if [[ "$DEBUG_REPERTOIRES" -gt 0 ]]; then
    debug_args=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")
  fi
  run_cmd "$log" \
    "$PYTHON_BIN" -u -m evals.sequencing_depth_experiment \
      --model ensemble_xgboost \
      --target_disease "$disease" \
      --metadata_path "$METADATA" \
      --repertoire_data_dir "$REPERTOIRE_DIR" \
      --depth_indices "$DEPTH_INDICES" \
      --xgboost_n_jobs "$SCALING_N_JOBS" \
      --xgboost_device cpu \
      --output_json "$output" \
      "${debug_args[@]}"
  local status=$?
  echo "[$(date +%T)] done scaling $disease exit=$status log=$log"
  return "$status"
}

run_driver_train_one() {
  local disease="$1"
  local slot="$2"
  local log="${DRIVER_LOGDIR}/ensemble_xgboost_driver_retrain_${disease}_${RUN_TS}.log"
  local output="${RESULTS}/ensemble_xgboost_driver_retrain_${disease}_${RUN_TS}_classification.csv"

  echo "[$(date +%T)] start driver retrain $disease slot=$slot"
  local debug_args=()
  if [[ "$DEBUG_REPERTOIRES" -gt 0 ]]; then
    debug_args=(--debug_repertoires "$DEBUG_REPERTOIRES")
  fi
  run_cmd "$log" \
    "$PYTHON_BIN" -u -m evals.ensemble_xgboost_disease_classification \
      --metadata_path "$METADATA" \
      --repertoire_data_dir "$REPERTOIRE_DIR" \
      --target_disease "$disease" \
      --n_jobs "$DRIVER_TRAIN_N_JOBS" \
      --xgb_device cpu \
      --output_csv "$output" \
      --model_save_dir "$DRIVER_MODEL_DIR" \
      "${debug_args[@]}"
  local status=$?
  echo "[$(date +%T)] done driver retrain $disease exit=$status log=$log"
  return "$status"
}

run_driver_score_one() {
  local disease="$1"
  local slot="$2"
  local log="${DRIVER_LOGDIR}/ensemble_xgboost_driver_${disease}_k${DRIVER_K_TAG}_${RUN_TS}.log"
  local output="${RESULTS}/ensemble_xgboost_driver_${disease}_k${DRIVER_K_TAG}_${RUN_TS}.csv"

  echo "[$(date +%T)] start driver scoring $disease slot=$slot"
  run_cmd "$log" \
    "$PYTHON_BIN" -u -m evals.ensemble_xgboost_driver_identification \
      --metadata_path "$METADATA" \
      --repertoire_data_dir "$REPERTOIRE_DIR" \
      --target_disease "$disease" \
      --driver_seqs_path "$DRIVER_SEQS" \
      --k "$DRIVER_K" \
      --model_save_dir "$DRIVER_MODEL_DIR" \
      --output_csv "$output" \
      "${DRIVER_MAX_REPERTOIRES[@]}"
  local status=$?
  echo "[$(date +%T)] done driver scoring $disease exit=$status log=$log"
  return "$status"
}

overall_status=0

if $RUN_DEMOGRAPHIC; then
  demographic_specs=(
    "adjust:Lupus"
    "adjust:HIV"
    "adjust:Influenza"
    "adjust:Covid19"
    "baseline:Lupus"
    "baseline:HIV"
    "baseline:Influenza"
    "baseline:Covid19"
  )
  echo "[$(date +%T)] Starting demographic phase (${#demographic_specs[@]} jobs, parallel=$DEMO_PARALLEL, n_jobs/job=$DEMO_N_JOBS)"
  run_pool "$DEMO_PARALLEL" run_demographic_one "${demographic_specs[@]}" || overall_status=1
fi

if $RUN_SCALING; then
  scaling_diseases=("Lupus" "HIV")
  echo "[$(date +%T)] Starting scaling phase (${#scaling_diseases[@]} jobs, parallel=$SCALING_PARALLEL, n_jobs/job=$SCALING_N_JOBS)"
  echo "[$(date +%T)] Scaling depths come from $DEPTH_INDICES: 1000, 5000, 10000, 25000, 50000, 75000 with 5 repeats"
  run_pool "$SCALING_PARALLEL" run_scaling_one "${scaling_diseases[@]}" || overall_status=1
fi

if $RUN_DRIVER; then
  driver_diseases=("Influenza" "Covid19")
  echo "[$(date +%T)] Starting driver retrain phase (${#driver_diseases[@]} jobs, parallel=$DRIVER_TRAIN_PARALLEL, n_jobs/job=$DRIVER_TRAIN_N_JOBS)"
  run_pool "$DRIVER_TRAIN_PARALLEL" run_driver_train_one "${driver_diseases[@]}" || overall_status=1

  if [[ "$overall_status" -eq 0 ]]; then
    echo "[$(date +%T)] Starting driver scoring phase (${#driver_diseases[@]} jobs, parallel=$DRIVER_SCORE_PARALLEL)"
    run_pool "$DRIVER_SCORE_PARALLEL" run_driver_score_one "${driver_diseases[@]}" || overall_status=1
  else
    echo "[$(date +%T)] Skipping driver scoring because retraining failed." >&2
  fi
fi

echo "[$(date +%T)] Done. status=$overall_status"
echo "Logs:"
echo "  demographic: $DEMO_LOGDIR"
echo "  scaling:     $SCALING_LOGDIR"
echo "  driver:      $DRIVER_LOGDIR"
echo "Model dirs:"
echo "  demographic: $DEMO_MODEL_DIR"
echo "  driver:      $DRIVER_MODEL_DIR"
exit "$overall_status"
