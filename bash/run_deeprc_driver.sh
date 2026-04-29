#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
K="100,1000,10000"

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --k=*) K="${arg#*=}" ;;
  esac
done

# ---- config ----
GPUS=(0)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DRIVER_SEQS=${REPO_ROOT}/data/public_clones/vdjdb_matches_expanded.csv
MODEL_SAVE_DIR=${REPO_ROOT}/results/deeprc_best
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeprc_driver
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Covid19")
  MAX_REPERTOIRES=20
else
  DISEASES=("HIV" "Influenza" "Covid19")
  MAX_REPERTOIRES=""
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
    log="${LOGDIR}/deeprc_driver_${disease}_k${k_tag}_${RUN_TS}.log"
    echo "[$(date +%T)] start $disease (k=${K}) on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease k=${K} on GPU $gpu"

      debug_flags=()
      [ -n "$MAX_REPERTOIRES" ] && debug_flags+=(--max_repertoires "$MAX_REPERTOIRES")

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.deeprc_2020_driver_identification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --driver_seqs_path "$DRIVER_SEQS" \
        --k "$K" \
        --model_save_dir "$MODEL_SAVE_DIR" \
        --output_csv "${RESULTS}/deeprc_2020_${disease}_driver_k${k_tag}_${RUN_TS}.csv" \
        --batch_size 32 \
        "${debug_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease k=${K} on GPU $gpu (exit $status)"
      exit $status
    } >"$log" 2>&1

    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $disease k=${K} on GPU $gpu | log: $log"
  } &
done

wait
echo "All jobs complete."
