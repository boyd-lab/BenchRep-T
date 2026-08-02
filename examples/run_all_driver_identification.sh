#!/usr/bin/env bash
set -euo pipefail

# Run the driver-sequence identification task through all six supported methods.
# Ground truth is the published VDJdb/Minervina match table, which is keyed to
# Mal-ID repertoire identifiers, so this task covers the Mal-ID cohort and the
# three diseases with antigen-specific annotations. Repertoires remain in the
# downloaded Hugging Face tree; staging creates only normalized metadata and
# symlinks. Run from any directory.
#
# Ensemble XGBoost only scores CDR3s with the models written by disease
# classification and never fits here, so it needs DISEASE_RUN_ROOT: the
# OUTPUT_ROOT of an earlier run_all_disease_classification.sh run.
#
#   OUTPUT_ROOT=results/disease bash examples/run_all_disease_classification.sh
#   DISEASE_RUN_ROOT=results/disease bash examples/run_all_driver_identification.sh
#
# ABMIL, DeepRC, and GIANA reuse that run's checkpoints and cluster files when
# they are present, and otherwise fit their own. Emerson and Ensemble Regression
# always refit. Without DISEASE_RUN_ROOT every method except Ensemble XGBoost
# runs self-contained, and Ensemble XGBoost is reported as skipped.

ROOT="${BENCHREP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/all_driver_identification/${RUN_ID}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}/staged_data}"
DRIVER_SEQS="${DRIVER_SEQS:-${DATA_ROOT}/Mal-ID/vdjdb_minervina_driver_seq_matches.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA="${USE_CONDA:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_GPU="${USE_GPU:-1}"
N_THREADS="${N_THREADS:-8}"
K="${K:-100,1000,10000}"
MAX_REPERTOIRES="${MAX_REPERTOIRES:-}"
DISEASE_RUN_ROOT="${DISEASE_RUN_ROOT:-}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-1}"
RANDOM_BASELINE="${RANDOM_BASELINE:-1}"
N_TRIALS="${N_TRIALS:-10000}"
RANDOM_SEED="${RANDOM_SEED:-7}"
METHODS="${METHODS:-emerson ensemble_regression ensemble_xgboost abmil deeprc giana}"
DISEASES="${DISEASES:-HIV Influenza Covid19}"

COHORT=malid

declare -A MODULES=(
  [emerson]=evals.emerson_2017_driver_identification
  [ensemble_regression]=evals.ensemble_regression_driver_identification
  [ensemble_xgboost]=evals.ensemble_xgboost_driver_identification
  [abmil]=evals.ensemble_abmil_driver_identification
  [deeprc]=evals.deeprc_2020_driver_identification
  [giana]=evals.giana_2021_driver_identification
)
declare -A ENVIRONMENTS=(
  [emerson]=benchrep-base
  [ensemble_regression]=benchrep-base
  [ensemble_xgboost]=benchrep-base
  [abmil]=abmil
  [deeprc]=deeprc
  [giana]=giana
)
# Methods that can consume the artifacts of a disease-classification run, and
# the flag each one uses. The artifact directory is always
# ${DISEASE_RUN_ROOT}/malid/<disease>/<method>/models.
declare -A REUSE_FLAGS=(
  [ensemble_xgboost]=--model_save_dir
  [abmil]=--model_save_dir
  [deeprc]=--model_save_dir
  [giana]=--cluster_dir
)
# Diseases covered by the published driver match table.
declare -A SUPPORTED_DISEASES=(
  [HIV]=1
  [Influenza]=1
  [Covid19]=1
)
# Only these evaluators accept --max_repertoires; the others ignore the knob.
declare -A SUBSAMPLE_METHODS=(
  [ensemble_xgboost]=1
  [abmil]=1
  [deeprc]=1
)

# Emerson and Ensemble Regression take space-separated integers for --k; the
# rest take one comma-separated string.
K_LIST=()
read -r -a K_LIST <<< "${K//,/ }"
read -r -a DISEASE_LIST <<< "${DISEASES}"

python_command() {
  local method=$1
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' "${CONDA_EXE}" run -n "${ENVIRONMENTS[$method]}" python
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

# Echo the disease-classification artifact directory for a method/disease pair,
# or nothing when reuse is off, unavailable, or not applicable to the method.
reuse_dir() {
  local method=$1 disease_tag=$2
  [[ "${REUSE_CHECKPOINTS}" == "1" ]] || return 0
  [[ -n "${REUSE_FLAGS[$method]+x}" ]] || return 0
  [[ -n "${DISEASE_RUN_ROOT}" ]] || return 0
  local candidate="${DISEASE_RUN_ROOT}/${COHORT}/${disease_tag}/${method}/models"
  if [[ -d "${candidate}" || "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${candidate}"
  fi
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download=(
    "${PYTHON_BIN}" -m utils.huggingface_data
    --output-dir "${DATA_ROOT}" --task drivers
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    download+=(--dry-run)
  fi
  (cd "${ROOT}" && "${download[@]}")
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  (cd "${ROOT}" && "${PYTHON_BIN}" -m utils.stage_disease_data \
    --data-root "${DATA_ROOT}" --output-root "${STAGE_ROOT}" --cohort "${COHORT}")
fi

metadata="${STAGE_ROOT}/${COHORT}/metadata.tsv"
repertoires="${STAGE_ROOT}/${COHORT}/repertoires"

for disease in "${DISEASE_LIST[@]}"; do
  if [[ -z "${SUPPORTED_DISEASES[$disease]+x}" ]]; then
    echo "Unknown driver-identification disease: ${disease}" >&2
    exit 2
  fi
done
for method in ${METHODS}; do
  if [[ -z "${MODULES[$method]+x}" ]]; then
    echo "Unknown method: ${method}" >&2
    exit 2
  fi
done

# Random-chance recall@k reference line for the driver-identification figure.
if [[ "${RANDOM_BASELINE}" == "1" ]]; then
  mapfile -t baseline_launcher < <(python_command emerson)
  baseline=(
    "${baseline_launcher[@]}" -u -m evals.compute_random_baseline_recall
    --metadata_path "${metadata}"
    --repertoire_dir "${repertoires}"
    --driver_seqs_path "${DRIVER_SEQS}"
    --diseases "${DISEASE_LIST[@]}"
    --ks "${K_LIST[@]}"
    --n_trials "${N_TRIALS}"
    --seed "${RANDOM_SEED}"
    --output_csv "${OUTPUT_ROOT}/random_baseline_recall.csv"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'BASELINE:'
    printf ' %q' "${baseline[@]}"
    printf '\n'
  else
    echo "[$(date --iso-8601=seconds)] random-chance baseline (k=${K})"
    (cd "${ROOT}" && "${baseline[@]}") 2>&1 \
      | tee "${OUTPUT_ROOT}/random_baseline_recall.log"
  fi
fi

run_count=0
skip_count=0
for disease in "${DISEASE_LIST[@]}"; do
  disease_tag=${disease// /_}
  for method in ${METHODS}; do
    method_root="${OUTPUT_ROOT}/${COHORT}/${disease_tag}/${method}"
    checkpoint=$(reuse_dir "${method}" "${disease_tag}")

    if [[ "${method}" == "ensemble_xgboost" && -z "${checkpoint}" ]]; then
      echo "SKIP ${disease} / ${method}: this method only scores with the models" \
        "saved by examples/run_all_disease_classification.sh. Run it first, then" \
        "set DISEASE_RUN_ROOT to its OUTPUT_ROOT with REUSE_CHECKPOINTS=1" \
        "(expected ${DISEASE_RUN_ROOT:-<DISEASE_RUN_ROOT>}/${COHORT}/${disease_tag}/${method}/models)." >&2
      skip_count=$((skip_count + 1))
      continue
    fi

    mapfile -t launcher < <(python_command "${method}")
    command=()
    if [[ "${method}" == "abmil" && "${USE_GPU}" != "1" ]]; then
      # The ABMIL driver evaluator has no CPU flag; hide the devices instead.
      command+=(env CUDA_VISIBLE_DEVICES=)
    fi
    command+=(
      "${launcher[@]}" -u -m "${MODULES[$method]}"
      --metadata_path "${metadata}"
      --repertoire_data_dir "${repertoires}"
      --target_disease "${disease}"
      --driver_seqs_path "${DRIVER_SEQS}"
      --output_csv "${method_root}/scores.csv"
    )

    case "${method}" in
      emerson|ensemble_regression)
        command+=(--k "${K_LIST[@]}")
        ;;
      ensemble_xgboost|abmil)
        command+=(--k "${K}")
        ;;
      deeprc)
        command+=(--k "${K}" --results_dir "${method_root}/models" --batch_size 32)
        [[ "${USE_GPU}" == "1" ]] && command+=(--device cuda:0) || command+=(--device cpu)
        ;;
      giana)
        command+=(
          --k "${K}" --results_dir "${method_root}/clusters"
          --n_threads "${N_THREADS}" --exact --threshold_iso 7
        )
        [[ "${USE_GPU}" == "1" ]] && command+=(--use_gpu)
        ;;
    esac

    # Every reuse flag is read-only: the evaluators load from it and fall back
    # to fitting in-place, so it is only passed when the artifacts exist.
    if [[ -n "${checkpoint}" ]]; then
      command+=("${REUSE_FLAGS[$method]}" "${checkpoint}")
    fi
    if [[ -n "${MAX_REPERTOIRES}" && -n "${SUBSAMPLE_METHODS[$method]+x}" ]]; then
      command+=(--max_repertoires "${MAX_REPERTOIRES}")
    fi

    run_count=$((run_count + 1))
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'RUN %03d:' "${run_count}"
      printf ' %q' "${command[@]}"
      printf '\n'
    else
      mkdir -p "${method_root}"
      echo "[$(date --iso-8601=seconds)] ${COHORT} / ${disease} / ${method} (k=${K})"
      (cd "${ROOT}" && "${command[@]}") 2>&1 | tee "${method_root}/run.log"
    fi
  done
done

echo "Prepared ${run_count} driver-identification runs under ${OUTPUT_ROOT}"
if [[ "${skip_count}" -gt 0 ]]; then
  echo "Skipped ${skip_count} runs that require saved disease-classification models"
fi
