#!/bin/bash
set -uo pipefail

# CPU-only Ensemble XGBoost scaling runner for depth 75000.
# Runs Lupus and HIV in parallel and automatically uses all available CPU cores.

DRY_RUN=false
SMOKE_TEST=false

SCALING_DEPTH=${SCALING_DEPTH:-75000}
SCALING_PARALLEL=${SCALING_PARALLEL:-2}
SCALING_N_JOBS=${SCALING_N_JOBS:-auto}
DEBUG_REPERTOIRES=0

for arg in "$@"; do
  case "$arg" in
    --dry_run)
      DRY_RUN=true
      ;;
    --smoke_test)
      SMOKE_TEST=true
      ;;
    --scaling_parallel=*)
      SCALING_PARALLEL="${arg#*=}"
      ;;
    --scaling_n_jobs=*)
      SCALING_N_JOBS="${arg#*=}"
      ;;
    --debug_repertoires=*)
      DEBUG_REPERTOIRES="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

get_total_cores() {
  if [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    echo "$SLURM_CPUS_ON_NODE"
  elif [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "$SLURM_CPUS_PER_TASK"
  else
    nproc
  fi
}

TOTAL_CORES=$(get_total_cores)

if $SMOKE_TEST; then
  SCALING_PARALLEL=1
  SCALING_N_JOBS=1
  DEBUG_REPERTOIRES="${DEBUG_REPERTOIRES:-6}"

  if [[ "$DEBUG_REPERTOIRES" -eq 0 ]]; then
    DEBUG_REPERTOIRES=6
  fi
else
  if [[ "$SCALING_N_JOBS" == "auto" ]]; then
    SCALING_N_JOBS=$(( TOTAL_CORES / SCALING_PARALLEL ))

    if [[ "$SCALING_N_JOBS" -lt 1 ]]; then
      SCALING_N_JOBS=1
    fi
  fi
fi

# Prevent thread oversubscription
export OMP_NUM_THREADS="$SCALING_N_JOBS"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

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

FILTERED_DEPTH_INDICES="${RESULTS}/depth_indices_max75k_depth${SCALING_DEPTH}_${RUN_TS}.json.gz"

prepare_scaling_depth_indices() {
  if $DRY_RUN; then
    echo "[$(date +%T)] DRY RUN: would write filtered depth indices for depth ${SCALING_DEPTH} to $FILTERED_DEPTH_INDICES"
    return 0
  fi

  "$PYTHON_BIN" - "$DEPTH_INDICES" "$FILTERED_DEPTH_INDICES" "$SCALING_DEPTH" <<'PY'
import gzip
import json
import sys

src, dst, depth_arg = sys.argv[1:]
keep = [int(depth_arg)]

with (gzip.open(src, "rt", encoding="utf-8") if src.endswith(".gz")
      else open(src, "r", encoding="utf-8")) as f:
    data = json.load(f)

available = set(data.get("depths", []))
missing = [d for d in keep if d not in available]

if missing:
    raise SystemExit(f"Requested depths not present in {src}: {missing}")

data["depths"] = keep

with gzip.open(dst, "wt", encoding="utf-8") as f:
    json.dump(data, f)
PY

  echo "[$(date +%T)] Wrote filtered depth indices for depth ${SCALING_DEPTH}: $FILTERED_DEPTH_INDICES"
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

  local log="${SCALING_LOGDIR}/ensemble_xgboost_scaling_${disease}_depth${SCALING_DEPTH}_${RUN_TS}.log"

  local output="${RESULTS}/ensemble_xgboost_${disease}_scaling_depth${SCALING_DEPTH}_${RUN_TS}.json"

  echo "[$(date +%T)] start scaling $disease depth=$SCALING_DEPTH slot=$slot"
  echo "[$(date +%T)] LOG_FILE ${disease} depth=${SCALING_DEPTH}: $log"

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
      --depth_indices "$FILTERED_DEPTH_INDICES" \
      --xgboost_n_jobs "$SCALING_N_JOBS" \
      --xgboost_device cpu \
      --output_json "$output" \
      ${debug_args[@]+"${debug_args[@]}"}

  local status=$?

  echo "[$(date +%T)] done scaling $disease depth=$SCALING_DEPTH exit=$status log=$log"

  return "$status"
}

overall_status=0
scaling_diseases=("Lupus" "HIV")

prepare_scaling_depth_indices || overall_status=1

echo "[$(date +%T)] Starting depth ${SCALING_DEPTH} scaling"
echo "[$(date +%T)] total_cores=$TOTAL_CORES parallel=$SCALING_PARALLEL n_jobs/job=$SCALING_N_JOBS estimated_threads=$((SCALING_PARALLEL * SCALING_N_JOBS))"

if [[ "$overall_status" -eq 0 ]]; then
  run_pool "$SCALING_PARALLEL" run_scaling_one "${scaling_diseases[@]}" || overall_status=1
fi

echo "[$(date +%T)] Done. status=$overall_status"
echo "Logs:"
echo "  scaling:     $SCALING_LOGDIR"

exit "$overall_status"