#!/bin/bash
set -uo pipefail

# ---- flags ----

DEBUG=false
DEBUG_REPERTOIRES=10
SUBMODEL="kmer_only"
ADJUST_DISTRIBUTION=false
N_JOBS=10

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_regression
mkdir -p "$LOGDIR" "$RESULTS"

DISEASES=(Lupus T1D HIV Influenza Covid19)
KMER_SIZES=(3 4 5)

# Gapped setting: true = plain k-mers + gapped variants; false = plain k-mers only
GAP_SETTINGS=(true false)

# FIFO concurrency token pool (N_JOBS parallel slots)
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for i in $(seq 1 "$N_JOBS"); do echo "$i" >&3; done

cd "${REPO_ROOT}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

for use_gaps in "${GAP_SETTINGS[@]}"; do
  for kmer_size in "${KMER_SIZES[@]}"; do
    for disease in "${DISEASES[@]}"; do
      read -r slot <&3  # blocks until a slot is free

      {
        # Build suffix encoding the key flags for use in file names
        suffix="${SUBMODEL}"
        $ADJUST_DISTRIBUTION && suffix+="_distadj"
        if $use_gaps; then suffix+="_gaps"; else suffix+="_nogaps"; fi
        suffix+="_${kmer_size}mer"

        log="${LOGDIR}/ensemble_regression_${suffix}_${disease}_${RUN_TS}.log"
        echo "[$(date +%T)] start $disease (submodel=$SUBMODEL, kmer=${kmer_size}, gaps=$use_gaps, slot=$slot) -> $log"

        {
          echo "[$(date +%T)] start $disease submodel=$SUBMODEL kmer_size=$kmer_size use_gaps=$use_gaps"

          extra_flags=()
          $ADJUST_DISTRIBUTION && extra_flags+=(--adjust_distribution_by_demographics)
          $use_gaps            || extra_flags+=(--no_gaps)
          if $DEBUG; then
            extra_flags+=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")
          fi

          python -u -m evals.ensemble_regression_disease_classification \
            --metadata_path "$METADATA" \
            --repertoire_data_dir "$REPERTOIRE_DIR" \
            --target_disease "$disease" \
            --submodel "$SUBMODEL" \
            --kmer_size "$kmer_size" \
            --output_csv "${RESULTS}/ensemble_regression_${suffix}_${disease}_${RUN_TS}_classification.csv" \
            "${extra_flags[@]}"

          status=$?
          echo "[$(date +%T)] done  $disease submodel=$SUBMODEL kmer_size=$kmer_size use_gaps=$use_gaps (exit $status)"
          exit $status
        } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

        echo "$slot" >&3  # return concurrency token
        echo "[$(date +%T)] done  $disease kmer=${kmer_size} gaps=$use_gaps | log: $log"
      } &
    done
  done
done

wait
echo "All jobs complete."
