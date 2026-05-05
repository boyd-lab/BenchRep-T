#!/bin/bash
set -uo pipefail

# CPU-only Ensemble XGBoost scaling runner for depth 75000.
# Runs Lupus and HIV in parallel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALING_DEPTH=75000 exec "${SCRIPT_DIR}/run_xgboost_scaling_50k.sh" "$@"
