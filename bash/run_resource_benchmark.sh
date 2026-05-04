#!/bin/bash
set -euo pipefail

# Launch wall-time / peak-memory benchmarks for AIRR-Bench methods.
#
# Recommended headline benchmark:
#   bash bash/run_resource_benchmark.sh --budget full --disease Lupus
#
# Fast plumbing check:
#   bash bash/run_resource_benchmark.sh --budget smoke --methods deeprc abmil

REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${METADATA:-${REPO_ROOT}/data/malid_clean/metadata.tsv}
REPERTOIRE_DIR=${REPERTOIRE_DIR:-${REPO_ROOT}/data/malid_clean/TCR}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/results/resource_benchmarks}
LOGDIR=${LOGDIR:-${REPO_ROOT}/logs/resource_benchmarks}
GPU=${GPU:-0}
PYTHON=${PYTHON:-python3}

mkdir -p "$OUTPUT_DIR" "$LOGDIR"
cd "$REPO_ROOT"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG="${LOGDIR}/resource_benchmark_${RUN_ID}.log"

echo "Writing benchmark log to $LOG"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m evals.resource_benchmark \
  --metadata_path "$METADATA" \
  --repertoire_data_dir "$REPERTOIRE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --run_id "$RUN_ID" \
  --gpu "$GPU" \
  "$@" 2>&1 | tee "$LOG"
