#!/bin/bash
set -uo pipefail

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
LOGDIR=${REPO_ROOT}/logs/vdjdb
OUTPUT_DIR=${REPO_ROOT}/data/vdjdb_matches

mkdir -p "$LOGDIR" "$OUTPUT_DIR"

ts=$(date +%Y%m%d_%H%M%S)
log="${LOGDIR}/process_driver_sequences_${ts}.log"

echo "[$(date +%T)] Starting driver sequence processing -> $log"

{
  echo "[$(date +%T)] Running process_driver_sequences.py"
  conda run -n airr-bench python -u "${REPO_ROOT}/preprocessing/process_driver_sequences.py" \
    --output-dir "$OUTPUT_DIR"
  status=$?
  echo "[$(date +%T)] Done (exit $status)"
  exit $status
} 2>&1 | tee "$log"
