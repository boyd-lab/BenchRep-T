#!/bin/bash
set -uo pipefail

# Intended for a node with 1 H100 GPU.
# Allocation:
#   - xgboost baseline for Lupus, HIV, Influenza, Covid19
#   - xgboost adjust for Lupus, HIV, Influenza, Covid19
# Runs all eight xgboost jobs concurrently on GPU 0 by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GPUS="0"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
JOB_SPECS=(
  "xgboost:baseline:Lupus"
  "xgboost:baseline:HIV"
  "xgboost:baseline:Influenza"
  "xgboost:baseline:Covid19"
  "xgboost:adjust:Lupus"
  "xgboost:adjust:HIV"
  "xgboost:adjust:Influenza"
  "xgboost:adjust:Covid19"
)

source "${SCRIPT_DIR}/run_missing_demographic_subsample_common.sh"
main "$@"
