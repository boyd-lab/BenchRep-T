#!/bin/bash
set -uo pipefail

# ---- flags ----

DEBUG=false
DEBUG_REPERTOIRES=10
SUBMODEL="ensemble"
COVARIATE_ADJUST=false
ADJUST_DISTRIBUTION=false
N_JOBS=20

# K-mer settings: sizes to evaluate and whether to include gapped variants.
# Set USE_GAPS=true to re-enable single-position gapped k-mer variants.
KMER_SIZES=(3 4 5)
USE_GAPS=false

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

# Build a base suffix encoding the key flags for use in file names
base_suffix="${SUBMODEL}"
$COVARIATE_ADJUST    && base_suffix+="_covadj"
$ADJUST_DISTRIBUTION && base_suffix+="_distadj"
if $USE_GAPS; then base_suffix+="_gaps"; else base_suffix+="_nogaps"; fi

for kmer_size in "${KMER_SIZES[@]}"; do
  for disease in "${DISEASES[@]}"; do
    read -r slot <&3  # blocks until a slot is free

    {
      suffix="${base_suffix}_${kmer_size}mer"
      ts=$(date +%Y%m%d_%H%M%S)
      log="${LOGDIR}/ensemble_regression_${suffix}_${disease}_${ts}.log"
      echo "[$(date +%T)] start $disease (submodel=$SUBMODEL, kmer=${kmer_size}, slot=$slot) -> $log"

      {
        echo "[$(date +%T)] start $disease submodel=$SUBMODEL kmer_size=$kmer_size"

        extra_flags=()
        $COVARIATE_ADJUST    && extra_flags+=(--covariate_adjust)
        $ADJUST_DISTRIBUTION && extra_flags+=(--adjust_distribution_by_demographics)
        $USE_GAPS            || extra_flags+=(--no_gaps)
        if $DEBUG; then
          extra_flags+=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")
        fi

        python -u -m evals.ensemble_regression_disease_classification \
          --metadata_path "$METADATA" \
          --repertoire_data_dir "$REPERTOIRE_DIR" \
          --target_disease "$disease" \
          --submodel "$SUBMODEL" \
          --kmer_size "$kmer_size" \
          --output_csv "${RESULTS}/ensemble_regression_${suffix}_${disease}_classification.csv" \
          "${extra_flags[@]}"

        status=$?
        echo "[$(date +%T)] done  $disease submodel=$SUBMODEL kmer_size=$kmer_size (exit $status)"
        exit $status
      } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

      echo "$slot" >&3  # return concurrency token
      echo "[$(date +%T)] done  $disease kmer=${kmer_size} | log: $log"
    } &
  done
done

wait
echo "All jobs complete."
