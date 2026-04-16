#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
DEBUG_REPERTOIRES=10
N_THREADS=10
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    --n_threads=*) N_THREADS="${arg#*=}" ;;
  esac
done

# ---- config ----
GPUS=(0 3)
JOBS_PER_GPU=3  # concurrent jobs sharing each GPU
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

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do echo "$g" >&3; done
done

cd "${REPO_ROOT}"

for disease in "${DISEASES[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/giana_${disease}_${ts}.log"
    echo "[$(date +%T)] start $disease on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease on GPU $gpu"

      debug_flags=()
      $DEBUG && debug_flags=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.giana_2021_disease_classification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --results_dir "${RESULTS}/giana" \
        --n_threads "$N_THREADS" \
        --use_gpu \
        --max_seqs_per_specimen 10000 \
        --exact \
        --threshold_iso 7 \
        --output_csv "${RESULTS}/giana_2021_${disease}_classification.csv" \
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
