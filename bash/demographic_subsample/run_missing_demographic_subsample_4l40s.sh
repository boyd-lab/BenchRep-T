#!/bin/bash
set -uo pipefail

# Intended for a node with 4 L40S GPUs.
# Allocation:
#   - DeepRC baseline for Lupus, HIV, Influenza, Covid19
#   - DeepRC adjust for Covid19

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GPUS="0,1,2,3"
JOB_SPECS=(
  "deeprc:baseline:Lupus"
  "deeprc:baseline:HIV"
  "deeprc:baseline:Influenza"
  "deeprc:baseline:Covid19"
  "deeprc:adjust:Covid19"
)

source "${SCRIPT_DIR}/run_missing_demographic_subsample_common.sh"
main "$@"
