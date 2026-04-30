#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=true
K="100,1000,10000"
FEATURES="full"

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --k=*) K="${arg#*=}" ;;
    --features=*) FEATURES="${arg#*=}" ;;
  esac
done

# ---- config ----
GPUS=(2 3)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DRIVER_SEQS=${REPO_ROOT}/data/public_clones/vdjdb_matches_expanded.csv
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/abmil_driver
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Covid19")
  MAX_REPERTOIRES=20
  EPOCHS=5
  PATIENCE=3
else
  DISEASES=("Influenza" "Covid19")
  MAX_REPERTOIRES=""
  EPOCHS=100
  PATIENCE=10
fi

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do echo "$g" >&3; done

cd "${REPO_ROOT}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

for disease in "${DISEASES[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    k_tag="${K//,/_}"
    log="${LOGDIR}/abmil_driver_${disease}_${FEATURES}_k${k_tag}_${RUN_TS}.log"
    echo "[$(date +%T)] start $disease (features=${FEATURES}, k=${K}) on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease features=${FEATURES} k=${K} on GPU $gpu"

      debug_flags=()
      [ -n "$MAX_REPERTOIRES" ] && debug_flags+=(--max_repertoires "$MAX_REPERTOIRES")

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.ensemble_abmil_driver_identification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --driver_seqs_path "$DRIVER_SEQS" \
        --k "$K" \
        --features "$FEATURES" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --output_csv "${RESULTS}/abmil_${disease}_${FEATURES}_driver_k${k_tag}_${RUN_TS}.csv" \
        "${debug_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease features=${FEATURES} k=${K} on GPU $gpu (exit $status)"
      exit $status
    } >"$log" 2>&1

    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
  } &
done

wait
echo "All jobs complete."
