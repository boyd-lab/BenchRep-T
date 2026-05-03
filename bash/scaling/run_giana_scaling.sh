#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
DEBUG_REPERTOIRES=10
N_THREADS=10
DISEASES_ARG="HIV,Lupus"
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --debug_repertoires=*) DEBUG_REPERTOIRES="${arg#*=}" ;;
    --n_threads=*) N_THREADS="${arg#*=}" ;;
    --diseases=*) DISEASES_ARG="${arg#*=}" ;;
    --gpus=*) : ;;  # accepted for backward compatibility; ignored on CPU
    --depth_indices=*) DEPTH_INDICES_OVERRIDE="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DEPTH_INDICES=${REPO_ROOT}/data/depth_indices/depth_indices_max75k.json.gz
DEPTH_INDICES=${DEPTH_INDICES_OVERRIDE:-$DEPTH_INDICES}
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana_scaling
mkdir -p "$LOGDIR" "$RESULTS" "$RESULTS/giana_scaling"

if $DEBUG; then
  DISEASES=("Lupus")
else
  IFS=',' read -r -a DISEASES <<< "$DISEASES_ARG"
fi

cd "${REPO_ROOT}"

for disease in "${DISEASES[@]}"; do
  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/giana_scaling_${disease}_${ts}.log"
    echo "[$(date +%T)] start ${disease} on CPU -> $log"

    {
      echo "[$(date +%T)] start ${disease} on CPU"

      debug_flags=()
      $DEBUG && debug_flags=(--debug --debug_repertoires "$DEBUG_REPERTOIRES")

      python -u -m evals.sequencing_depth_experiment \
        --model giana_2021 \
        --target_disease "$disease" \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --depth_indices "$DEPTH_INDICES" \
        --giana_results_dir "${RESULTS}/giana_scaling/${disease}" \
        --giana_n_threads "$N_THREADS" \
        --giana_threshold_iso 7 \
        --giana_cpu \
        --output_json "${RESULTS}/giana_2021_${disease}_scaling.json" \
        ${debug_flags[@]+"${debug_flags[@]}"}

      status=$?
      echo "[$(date +%T)] done  ${disease} (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "[$(date +%T)] done  ${disease} | log: $log"
  } &
done

wait
echo "All jobs complete."
