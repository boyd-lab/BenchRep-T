#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
DEBUG_REPERTOIRES=10
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
  esac
done

# ---- config ----
GPUS=(0 1 2 3)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeptcr
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Lupus")
else
  DISEASES=("Lupus" "T1D" "HIV" "Influenza" "Covid19")
fi

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
    log="${LOGDIR}/deeptcr_${disease}_${ts}.log"
    echo "[$(date +%T)] start $disease on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease on GPU $gpu"

      debug_flags=()
      $DEBUG && debug_flags=(--debug --debug_repertoires "$DEBUG_REPERTOIRES" --epochs_max 5)

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.deeptcr_2021_disease_classification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --output_csv "${RESULTS}/deeptcr_2021_${disease}_classification.csv" \
        --results_dir "${RESULTS}/deeptcr" \
        --batch_size 4 \
        --device 0 \
        "${debug_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease on GPU $gpu (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
  } &
done

wait
echo "All jobs complete."
