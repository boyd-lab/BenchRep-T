#!/bin/bash
set -uo pipefail

# Intended for a node with 3 L40S GPUs.
# Allocation:
#   - DeepTCR baseline for Lupus, HIV, Influenza, Covid19

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_GPUS="0,1,3"
DEEPTCR_CONDA_ENV="${DEEPTCR_CONDA_ENV:-deeptcr}"
JOB_SPECS=(
  "deeptcr:baseline:Lupus"
  "deeptcr:baseline:HIV"
  "deeptcr:baseline:Influenza"
  "deeptcr:baseline:Covid19"
)

source "${SCRIPT_DIR}/run_missing_demographic_subsample_common.sh"
main "$@"
