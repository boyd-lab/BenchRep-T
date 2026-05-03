#!/bin/bash
set -uo pipefail

# ---- flags ----

DEBUG=false
DEBUG_REPERTOIRES=10
ADJUST_DISTRIBUTION=false
N_JOBS=10

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
MODEL_SAVE_DIR=${REPO_ROOT}/results/ensemble_xgboost_models
LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost
mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

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

suffix="ensemble_xgboost"
$ADJUST_DISTRIBUTION && suffix+="_distadj"

for disease in "${DISEASES[@]}"; do
  read -r slot <&3

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/ensemble_xgboost_${suffix}_${disease}_${ts}.log"
    echo "[$(date +%T)] start $disease (slot=$slot) -> $log"

    {
      echo "[$(date +%T)] start $disease"

      extra_flags=()
      $ADJUST_DISTRIBUTION && extra_flags+=(--adjust_distribution_by_demographics)
      if $DEBUG; then
        extra_flags+=(--debug_repertoires "$DEBUG_REPERTOIRES")
      fi

      python -u -m evals.ensemble_xgboost_disease_classification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --output_csv "${RESULTS}/ensemble_xgboost_${suffix}_${disease}_classification.csv" \
        --model_save_dir "$MODEL_SAVE_DIR" \
        "${extra_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "$slot" >&3
    echo "[$(date +%T)] done  $disease | log: $log"
  } &
done

wait
echo "All jobs complete."
