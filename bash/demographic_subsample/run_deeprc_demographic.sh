#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
N_UPDATES=10000
EVALUATE_AT=1000
BATCH_SIZE=32
SAMPLE_N_SEQUENCES=10000
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --n_updates=*) N_UPDATES="${arg#*=}" ;;
    --evaluate_at=*) EVALUATE_AT="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --sample_n_sequences=*) SAMPLE_N_SEQUENCES="${arg#*=}" ;;
  esac
done

MODES=("adjust" "baseline")

# ---- config ----
GPUS=(0 1)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeprc_demographic
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Lupus")
  N_UPDATES=100
  EVALUATE_AT=50
else
  DISEASES=("Lupus" "HIV" "Influenza" "Covid19")
fi

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do echo "$g" >&3; done

cd "${REPO_ROOT}"

RUN_TS=$(date +%Y%m%d_%H%M%S)

for mode in "${MODES[@]}"; do
  for disease in "${DISEASES[@]}"; do
    read -r gpu <&3

    {
      ts=$(date +%Y%m%d_%H%M%S)
      log="${LOGDIR}/deeprc_demographic_${mode}_${disease}_${ts}.log"
      echo "[$(date +%T)] start $disease on GPU $gpu (mode=$mode) -> $log"

      {
        echo "[$(date +%T)] start $disease on GPU $gpu mode=$mode"

        extra_flags=()
        if [[ "$mode" == "baseline" ]]; then
          extra_flags+=(--random_baseline_seeds 7 14 21 28 35)
        else
          extra_flags+=(--adjust_distribution_by_demographics)
        fi

        CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.deeprc_2020_disease_classification \
          --metadata_path "$METADATA" \
          --repertoire_data_dir "$REPERTOIRE_DIR" \
          --target_disease "$disease" \
          --output_csv "${RESULTS}/deeprc_2020_${mode}_${disease}_${RUN_TS}_classification.csv" \
          --results_dir "${RESULTS}/deeprc_demographic" \
          --device "cuda:0" \
          --batch_size "$BATCH_SIZE" \
          --n_updates "$N_UPDATES" \
          --evaluate_at "$EVALUATE_AT" \
          --sample_n_sequences "$SAMPLE_N_SEQUENCES" \
          "${extra_flags[@]}"

        status=$?
        echo "[$(date +%T)] done  $disease on GPU $gpu mode=$mode (exit $status)"
        exit $status
      } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

      echo "$gpu" >&3
      echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
    } &
  done
done

wait
echo "All jobs complete."
