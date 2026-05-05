#!/bin/bash
set -uo pipefail

# CPU-only runner for remaining Ensemble XGBoost scaling experiments.
#
# Experiments:
#   Sequencing-depth scaling:
#      - Lupus, HIV using data/depth_indices/depth_indices_max75k.json.gz
#      - depths: 50000, 75000; 5 repeats

DRY_RUN=false
SMOKE_TEST=false

SCALING_PARALLEL=2
SCALING_N_JOBS=30
DEBUG_REPERTOIRES=0
SCALING_DEPTHS=(50000 75000)
SCALING_DEPTH_TAG="50000_75000"

for arg in "$@"; do
  case "$arg" in
    --dry_run)
      DRY_RUN=true
      ;;
    --smoke_test)
      SMOKE_TEST=true
      ;;
    --scaling_parallel=*) SCALING_PARALLEL="${arg#*=}" ;;
    --scaling_n_jobs=*) SCALING_N_JOBS="${arg#*=}" ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if $SMOKE_TEST; then
  SCALING_PARALLEL=1
  SCALING_N_JOBS=1
  DEBUG_REPERTOIRES="${DEBUG_REPERTOIRES:-6}"
  if [[ "$DEBUG_REPERTOIRES" -eq 0 ]]; then
    DEBUG_REPERTOIRES=6
  fi
  SCALING_DEPTHS=(25000)
  SCALING_DEPTH_TAG="25000"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DEPTH_INDICES=${REPO_ROOT}/data/depth_indices/depth_indices_max75k.json.gz
RESULTS=${REPO_ROOT}/results
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_TS=$(date +%Y%m%d_%H%M%S)

SCALING_LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_scaling_cpu

mkdir -p "$RESULTS" "$SCALING_LOGDIR"

cd "$REPO_ROOT" || exit 1

prepare_scaling_depth_indices() {
  local filtered="${RESULTS}/depth_indices_max75k_depths_${SCALING_DEPTH_TAG}_${RUN_TS}.json.gz"

  if $DRY_RUN; then
    DEPTH_INDICES="$filtered"
    echo "[$(date +%T)] DRY RUN: would write filtered depth indices to $DEPTH_INDICES"
    return 0
  fi

  "$PYTHON_BIN" - "$DEPTH_INDICES" "$filtered" "${SCALING_DEPTHS[@]}" <<'PY'
import gzip
import json
import sys

src, dst, *depth_args = sys.argv[1:]
keep = [int(x) for x in depth_args]

with (gzip.open(src, "rt", encoding="utf-8") if src.endswith(".gz") else open(src, "r", encoding="utf-8")) as f:
    data = json.load(f)

available = set(data.get("depths", []))
missing = [d for d in keep if d not in available]
if missing:
    raise SystemExit(f"Requested depths not present in {src}: {missing}")

data["depths"] = keep

with gzip.open(dst, "wt", encoding="utf-8") as f:
    json.dump(data, f)
PY

  DEPTH_INDICES="$filtered"
  echo "[$(date +%T)] Wrote filtered depth indices: $DEPTH_INDICES"
}

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

overall_status=0
scaling_diseases=("Lupus" "HIV")

prepare_scaling_depth_indices || overall_status=1
echo "[$(date +%T)] Starting scaling phase (${#scaling_diseases[@]} jobs, parallel=$SCALING_PARALLEL, n_jobs/job=$SCALING_N_JOBS)"
echo "[$(date +%T)] Scaling depths come from $DEPTH_INDICES: ${SCALING_DEPTHS[*]} with 5 repeats"
if [[ "$overall_status" -eq 0 ]]; then
  run_pool "$SCALING_PARALLEL" run_scaling_one "${scaling_diseases[@]}" || overall_status=1
fi

echo "[$(date +%T)] Done. status=$overall_status"
echo "Logs:"
echo "  scaling:     $SCALING_LOGDIR"
exit "$overall_status"
