#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=true
DEBUG_REPERTOIRES=10
N_THREADS=4
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    --n_threads=*) N_THREADS="${arg#*=}" ;;
  esac
done

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana
mkdir -p "$LOGDIR" "$RESULTS/giana"

if $DEBUG; then
  DISEASES=("Lupus")
else
  DISEASES=("Lupus" "T1D" "HIV" "Influenza" "Covid19")
fi

cd "${REPO_ROOT}"

for disease in "${DISEASES[@]}"; do
  ts=$(date +%Y%m%d_%H%M%S)
  log="${LOGDIR}/giana_${disease}_${ts}.log"
  echo "[$(date +%T)] start $disease -> $log"

  debug_flags=()
  $DEBUG && debug_flags=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")

  {
    python -u -m evals.giana_2021_disease_classification \
      --metadata_path "$METADATA" \
      --repertoire_data_dir "$REPERTOIRE_DIR" \
      --target_disease "$disease" \
      --results_dir "${RESULTS}/giana" \
      --n_threads "$N_THREADS" \
      --output_csv "${RESULTS}/giana_2021_${disease}_classification.csv" \
      "${debug_flags[@]}"
    echo "[$(date +%T)] done  $disease | log: $log"
  } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi &

done

wait
echo "All jobs complete."
