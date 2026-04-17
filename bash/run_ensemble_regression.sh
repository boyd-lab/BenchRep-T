#!/bin/bash
set -uo pipefail

# ---- flags ----

DEBUG=false
DEBUG_REPERTOIRES=10
SUBMODEL="ensemble"
COVARIATE_ADJUST=true
ADJUST_DISTRIBUTION=false
N_JOBS=5

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_regression
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Lupus")
else
  DISEASES=("Lupus" "T1D" "HIV" "Influenza" "Covid19")
fi

# FIFO concurrency token pool (N_JOBS parallel slots)
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for i in $(seq 1 "$N_JOBS"); do echo "$i" >&3; done

cd "${REPO_ROOT}"

# Build a suffix encoding the key flags for use in file names
suffix="${SUBMODEL}"
$COVARIATE_ADJUST      && suffix+="_covadj"
$ADJUST_DISTRIBUTION   && suffix+="_distadj"

for disease in "${DISEASES[@]}"; do
  read -r slot <&3  # blocks until a slot is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/ensemble_regression_${suffix}_${disease}_${ts}.log"
    echo "[$(date +%T)] start $disease (submodel=$SUBMODEL, slot=$slot) -> $log"

    {
      echo "[$(date +%T)] start $disease submodel=$SUBMODEL"

      extra_flags=()
      $COVARIATE_ADJUST && extra_flags+=(--covariate_adjust)
      $ADJUST_DISTRIBUTION && extra_flags+=(--adjust_distribution_by_demographics)
      if $DEBUG; then
        extra_flags+=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")
      fi

      python -u -m evals.ensemble_regression_disease_classification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --submodel "$SUBMODEL" \
        --output_csv "${RESULTS}/ensemble_regression_${suffix}_${disease}_classification.csv" \
        "${extra_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease submodel=$SUBMODEL (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "$slot" >&3  # return concurrency token
    echo "[$(date +%T)] done  $disease | log: $log"
  } &
done

wait
echo "All jobs complete."
