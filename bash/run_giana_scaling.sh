#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
  esac
done

# ---- config ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DEPTH_INDICES=${REPO_ROOT}/data/depth_indices/depth_indices_max75k.json.gz
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana_scaling
mkdir -p "$LOGDIR" "$RESULTS"

if $DEBUG; then
  DISEASES=("Lupus")
else
  DISEASES=("Lupus" "HIV")
fi

cd "${REPO_ROOT}"

for disease in "${DISEASES[@]}"; do
  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="${LOGDIR}/giana_scaling_${disease}_${ts}.log"
    echo "[$(date +%T)] start ${disease} -> $log"

    {
      echo "[$(date +%T)] start ${disease}"

      CUDA_VISIBLE_DEVICES=1 python -u -m evals.sequencing_depth_experiment \
        --model giana_2021 \
        --target_disease "$disease" \
        --metadata_path "$METADATA" \
        --repertoire_data_dir "$REPERTOIRE_DIR" \
        --depth_indices "$DEPTH_INDICES" \
        --output_json "${RESULTS}/giana_2021_${disease}_scaling.json"

      status=$?
      echo "[$(date +%T)] done  ${disease} (exit $status)"
      exit $status
    } 2>&1 | if $DEBUG; then tee "$log"; else cat >"$log"; fi

    echo "[$(date +%T)] done  ${disease} | log: $log"
  } &
done

wait
echo "All jobs complete."
