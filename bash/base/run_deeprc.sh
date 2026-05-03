#!/bin/bash
set -uo pipefail

# ---- config ----
GPUS=(0)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeprc
mkdir -p "$LOGDIR" "$RESULTS"

DISEASES=("Covid19")

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do echo "$g" >&3; done

cd "${REPO_ROOT}"

for disease in "${DISEASES[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/deeprc_${disease}_${ts}.log"
    echo "[$(date +%T)] start $disease on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease on GPU $gpu"

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.deeprc_2020_disease_classification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --output_csv "${RESULTS}/deeprc_2020_${disease}_classification.csv" \
        --batch_size 32

      status=$?
      echo "[$(date +%T)] done  $disease on GPU $gpu (exit $status)"
      exit $status
    } >"$log" 2>&1

    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
  } &
done

wait
echo "All jobs complete."
