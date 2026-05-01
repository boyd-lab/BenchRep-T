#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
DEBUG_REPERTOIRES=10
FEATURES="full"
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    --features=*) FEATURES="${arg#*=}" ;;
  esac
done

MODES=("adjust" "baseline")

# ---- config ----
GPUS=(2 3)
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/abmil_demographic
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Lupus")
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
    read -r gpu <&3  # blocks until a GPU is free

    {
      ts=$(date +%Y%m%d_%H%M%S)
      log="${LOGDIR}/abmil_demographic_${mode}_${disease}_${FEATURES}_${ts}.log"
      echo "[$(date +%T)] start $disease on GPU $gpu (mode=$mode) -> $log"

      {
        echo "[$(date +%T)] start $disease on GPU $gpu mode=$mode"

        extra_flags=()
        if $DEBUG; then
          extra_flags+=(--epochs 5 --max_repertoires_per_class "$DEBUG_REPERTOIRES")
        fi

        if [[ "$mode" == "baseline" ]]; then
          extra_flags+=(--random_baseline_seeds 7 14 21 28 35)
        else
          extra_flags+=(--adjust_distribution_by_demographics)
        fi

        CUDA_VISIBLE_DEVICES="$gpu" python -u -m evals.ensemble_abmil_disease_classification \
          --metadata_path "$METADATA" \
          --repertoire_data_dir "$REPERTOIRE_DIR" \
          --target_disease "$disease" \
          --features "$FEATURES" \
          --output_csv "${RESULTS}/abmil_${mode}_${disease}_${FEATURES}_${RUN_TS}_classification.csv" \
          "${extra_flags[@]}"

        status=$?
        echo "[$(date +%T)] done  $disease on GPU $gpu mode=$mode (exit $status)"
        exit $status
      } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

      echo "$gpu" >&3  # return GPU token
      echo "[$(date +%T)] done  $disease on GPU $gpu | log: $log"
    } &
  done
done

wait
echo "All jobs complete."
