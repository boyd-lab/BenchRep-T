#!/bin/bash
set -uo pipefail

cat >&2 <<'USAGE'
This launcher has been split into two single-depth scripts:

  bash bash/run_xgboost_scaling_50k.sh
  bash bash/run_xgboost_scaling_75k.sh

Each script runs Lupus and HIV in parallel for its depth.
USAGE

exit 2
