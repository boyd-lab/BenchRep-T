#!/bin/bash
set -uo pipefail

# CPU-only runner for Ensemble XGBoost driver sequence experiments.
#
# This uses a presaved base Ensemble XGBoost checkpoint for HIV and runs
# driver identification at k=100,1000,10000. It does not retrain models.

DRY_RUN=false
SMOKE_TEST=false

DRIVER_SCORE_PARALLEL=2
DEBUG_REPERTOIRES=0
DRIVER_K="100,1000,10000"
DRIVER_K_TAG="100_1000_10000"
DRIVER_MAX_REPERTOIRES=()

for arg in "$@"; do
  case "$arg" in
    --dry_run)
      DRY_RUN=true
      ;;
    --smoke_test)
      SMOKE_TEST=true
      ;;
    --driver_score_parallel=*) DRIVER_SCORE_PARALLEL="${arg#*=}" ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if $SMOKE_TEST; then
  DRIVER_SCORE_PARALLEL=1
  DEBUG_REPERTOIRES="${DEBUG_REPERTOIRES:-6}"
  if [[ "$DEBUG_REPERTOIRES" -eq 0 ]]; then
    DEBUG_REPERTOIRES=6
  fi
  DRIVER_K="1,5"
  DRIVER_K_TAG="1_5"
  DRIVER_MAX_REPERTOIRES=(--max_repertoires 12)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DRIVER_SEQS=${REPO_ROOT}/data/public_clones/vdjdb_matches_expanded.csv
RESULTS=${REPO_ROOT}/results
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_TS=$(date +%Y%m%d_%H%M%S)

DRIVER_LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_driver_cpu
DRIVER_MODEL_DIR=${DRIVER_MODEL_DIR:-${RESULTS}/ensemble_xgboost_models}

mkdir -p "$RESULTS" "$DRIVER_LOGDIR"

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

check_presaved_checkpoint() {
  local disease="$1"
  local missing=0
  local fold
  local required=(
    kmer_model.ubj
    kmer_vectorizer.pkl
    vj_model.ubj
    vj_vectorizer.pkl
    meta.json
  )

  for fold in 0 1 2; do
    local fold_dir="${DRIVER_MODEL_DIR}/${disease}/fold${fold}"
    if [[ ! -d "$fold_dir" ]]; then
      echo "Missing checkpoint fold directory: $fold_dir" >&2
      missing=1
      continue
    fi
    local filename
    for filename in "${required[@]}"; do
      if [[ ! -f "${fold_dir}/${filename}" ]]; then
        echo "Missing checkpoint file: ${fold_dir}/${filename}" >&2
        missing=1
      fi
    done
  done

  return "$missing"
}

overall_status=0
driver_diseases=("HIV")

echo "[$(date +%T)] Checking presaved HIV checkpoint in $DRIVER_MODEL_DIR"
check_presaved_checkpoint "HIV" || overall_status=1

if [[ "$overall_status" -eq 0 ]]; then
  echo "[$(date +%T)] Starting driver scoring phase (${#driver_diseases[@]} jobs, parallel=$DRIVER_SCORE_PARALLEL)"
  run_pool "$DRIVER_SCORE_PARALLEL" run_driver_score_one "${driver_diseases[@]}" || overall_status=1
else
  echo "[$(date +%T)] Skipping driver scoring because the HIV checkpoint is incomplete." >&2
fi

echo "[$(date +%T)] Done. status=$overall_status"
echo "Logs:"
echo "  driver: $DRIVER_LOGDIR"
echo "Model dirs:"
echo "  driver: $DRIVER_MODEL_DIR"
exit "$overall_status"
