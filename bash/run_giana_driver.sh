#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
DEBUG_REPERTOIRES=10
K="100,1000,10000"
N_THREADS=10

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    --k=*) K="${arg#*=}" ;;
    --n_threads=*) N_THREADS="${arg#*=}" ;;
  esac
done

# ---- config ----
GPUS=(0 3)
JOBS_PER_GPU=3  # concurrent jobs sharing each GPU
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DRIVER_SEQS=${REPO_ROOT}/data/public_clones/vdjdb_matches_expanded.csv
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana_driver
mkdir -p "$LOGDIR" "$RESULTS/giana_driver"

if $DEBUG; then
  DISEASES=("Covid19")
else
  DISEASES=("Influenza" "Covid19")
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

RUN_TS=$(date +%Y%m%d_%H%M%S)

for disease in "${DISEASES[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    k_tag="${K//,/_}"
    log="${LOGDIR}/giana_driver_${disease}_k${k_tag}_${RUN_TS}.log"
    echo "[$(date +%T)] start $disease (k=${K}) on GPU $gpu -> $log"

    {
      echo "[$(date +%T)] start $disease k=${K} on GPU $gpu"

      debug_flags=()
      $DEBUG && debug_flags=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")

      # Reuse cluster files from disease classification run if available
      cluster_dir_flag=()
      candidate="${RESULTS}/giana/${disease}_fold0/queryFinal.txt"
      if [ -f "$candidate" ]; then
        cluster_dir_flag=(--cluster_dir "${RESULTS}/giana")
      fi

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.giana_2021_driver_identification \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --target_disease "$disease" \
        --driver_seqs_path "$DRIVER_SEQS" \
        --k "$K" \
        --results_dir "${RESULTS}/giana_driver" \
        --n_threads "$N_THREADS" \
        --use_gpu \
        --max_seqs_per_specimen 10000 \
        --exact \
        --threshold_iso 7 \
        --output_csv "${RESULTS}/giana_2021_${disease}_driver_k${k_tag}_${RUN_TS}.csv" \
        "${cluster_dir_flag[@]}" \
        "${debug_flags[@]}"

      status=$?
      echo "[$(date +%T)] done  $disease k=${K} on GPU $gpu (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
  } &
done

wait
echo "All jobs complete."
