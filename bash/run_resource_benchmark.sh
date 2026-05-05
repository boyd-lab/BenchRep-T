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
PYTHON=${PYTHON:-python}
BENCHMARK_CONDA_ENV=${BENCHMARK_CONDA_ENV:-airr-bench}
DEEPRC_CONDA_ENV=${DEEPRC_CONDA_ENV:-deeprc}
DEEPTCR_CONDA_ENV=${DEEPTCR_CONDA_ENV:-deeptcr}
ABMIL_CONDA_ENV=${ABMIL_CONDA_ENV:-airr-bench}
GIANA_CONDA_ENV=${GIANA_CONDA_ENV:-giana}
AIRR_BENCH_CONDA_ENV=${AIRR_BENCH_CONDA_ENV:-airr-bench}
XGBOOST_CONDA_ENV=${XGBOOST_CONDA_ENV:-$AIRR_BENCH_CONDA_ENV}
REGRESSION_CONDA_ENV=${REGRESSION_CONDA_ENV:-$AIRR_BENCH_CONDA_ENV}
EMERSON_CONDA_ENV=${EMERSON_CONDA_ENV:-$AIRR_BENCH_CONDA_ENV}
OSTMEYER_CONDA_ENV=${OSTMEYER_CONDA_ENV:-$AIRR_BENCH_CONDA_ENV}

mkdir -p "$OUTPUT_DIR" "$LOGDIR"
cd "$REPO_ROOT"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG="${LOGDIR}/resource_benchmark_${RUN_ID}.log"

echo "Writing benchmark log to $LOG"

PY_CMD=("$PYTHON")
if [ -n "$BENCHMARK_CONDA_ENV" ]; then
  PY_CMD=(conda run --no-capture-output -n "$BENCHMARK_CONDA_ENV" "$PYTHON")
fi

export DEEPRC_CONDA_ENV DEEPTCR_CONDA_ENV ABMIL_CONDA_ENV GIANA_CONDA_ENV
export XGBOOST_CONDA_ENV REGRESSION_CONDA_ENV EMERSON_CONDA_ENV OSTMEYER_CONDA_ENV

CUDA_VISIBLE_DEVICES="$GPU" "${PY_CMD[@]}" -u -m evals.resource_benchmark \
  --metadata_path "$METADATA" \
  --repertoire_data_dir "$REPERTOIRE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --run_id "$RUN_ID" \
  --gpu "$GPU" \
  "$@" 2>&1 | tee "$LOG"
