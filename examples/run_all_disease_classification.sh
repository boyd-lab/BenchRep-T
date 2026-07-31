#!/usr/bin/env bash
set -euo pipefail

# Run every disease/cohort task through all eight disease-classification methods.
# Repertoires remain in the downloaded Hugging Face tree; staging creates only
# normalized metadata and symlinks. Run from any directory.

ROOT="${BENCHREP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/all_disease_classification/${RUN_ID}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}/staged_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA="${USE_CONDA:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_GPU="${USE_GPU:-1}"
N_JOBS="${N_JOBS:-8}"
N_THREADS="${N_THREADS:-8}"
MAX_FOLDS="${MAX_FOLDS:-}"
METHODS="${METHODS:-emerson ostmeyer ensemble_regression ensemble_xgboost abmil deeprc deeptcr giana}"
DATASETS="${DATASETS:-malid mitchell-t1d rawat-t1d tb ra cmv}"

declare -A MODULES=(
  [emerson]=evals.emerson_2017_disease_classification
  [ostmeyer]=evals.ostmeyer_2019_disease_classification
  [ensemble_regression]=evals.ensemble_regression_disease_classification
  [ensemble_xgboost]=evals.ensemble_xgboost_disease_classification
  [abmil]=evals.ensemble_abmil_disease_classification
  [deeprc]=evals.deeprc_2020_disease_classification
  [deeptcr]=evals.deeptcr_2021_disease_classification
  [giana]=evals.giana_2021_disease_classification
)
declare -A ENVIRONMENTS=(
  [emerson]=benchrep-base
  [ostmeyer]=benchrep-base
  [ensemble_regression]=benchrep-base
  [ensemble_xgboost]=benchrep-base
  [abmil]=abmil
  [deeprc]=deeprc
  [deeptcr]=deeptcr
  [giana]=giana
)
declare -A TARGETS=(
  [malid]="HIV,T1D,Lupus,Covid19,Influenza"
  [mitchell-t1d]="T1D"
  [rawat-t1d]="T1D"
  [tb]="Progressor"
  [ra]="Rheumatoid Arthritis"
  [cmv]="CMV"
)

python_command() {
  local method=$1
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' "${CONDA_EXE}" run -n "${ENVIRONMENTS[$method]}" python
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download=("${PYTHON_BIN}" -m utils.huggingface_data --output-dir "${DATA_ROOT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    download+=(--dry-run)
  fi
  (cd "${ROOT}" && "${download[@]}")
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  stage_args=(
    "${PYTHON_BIN}" -m utils.stage_disease_data
    --data-root "${DATA_ROOT}" --output-root "${STAGE_ROOT}"
  )
  for dataset in ${DATASETS}; do
    stage_args+=(--cohort "${dataset}")
  done
  (cd "${ROOT}" && "${stage_args[@]}")
fi

run_count=0
for dataset in ${DATASETS}; do
  if [[ -z "${TARGETS[$dataset]+x}" ]]; then
    echo "Unknown dataset: ${dataset}" >&2
    exit 2
  fi
  metadata="${STAGE_ROOT}/${dataset}/metadata.tsv"
  repertoires="${STAGE_ROOT}/${dataset}/repertoires"

  while IFS= read -r target; do
    [[ -z "${target}" ]] && continue
    target_tag=${target// /_}
    for method in ${METHODS}; do
      if [[ -z "${MODULES[$method]+x}" ]]; then
        echo "Unknown method: ${method}" >&2
        exit 2
      fi
      method_root="${OUTPUT_ROOT}/${dataset}/${target_tag}/${method}"
      mapfile -t launcher < <(python_command "${method}")
      command=(
        "${launcher[@]}" -u -m "${MODULES[$method]}"
        --metadata_path "${metadata}"
        --repertoire_data_dir "${repertoires}"
        --target_disease "${target}"
        --output_csv "${method_root}/scores.csv"
      )
      [[ -n "${MAX_FOLDS}" ]] && command+=(--max_folds "${MAX_FOLDS}")

      case "${method}" in
        emerson)
          command+=(--model_save_dir "${method_root}/models")
          ;;
        ostmeyer)
          command+=(--model_save_dir "${method_root}/models")
          ;;
        ensemble_regression)
          command+=(--model_save_dir "${method_root}/models" --n_jobs "${N_JOBS}")
          ;;
        ensemble_xgboost)
          command+=(--model_save_dir "${method_root}/models" --n_jobs "${N_JOBS}")
          [[ "${USE_GPU}" == "1" ]] && command+=(--xgb_device cuda)
          ;;
        abmil)
          command+=(--model_save_dir "${method_root}/models")
          [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
          ;;
        deeprc)
          command+=(--results_dir "${method_root}/models" --n_worker_processes "${N_JOBS}")
          [[ "${USE_GPU}" == "1" ]] && command+=(--device cuda:0) || command+=(--device cpu)
          ;;
        deeptcr)
          command+=(--results_dir "${method_root}/models" --n_jobs "${N_JOBS}")
          ;;
        giana)
          command+=(--results_dir "${method_root}/models" --n_threads "${N_THREADS}")
          [[ "${USE_GPU}" == "1" ]] && command+=(--use_gpu)
          ;;
      esac

      run_count=$((run_count + 1))
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'RUN %03d:' "${run_count}"
        printf ' %q' "${command[@]}"
        printf '\n'
      else
        mkdir -p "${method_root}"
        echo "[$(date --iso-8601=seconds)] ${dataset} / ${target} / ${method}"
        (cd "${ROOT}" && "${command[@]}") 2>&1 | tee "${method_root}/run.log"
      fi
    done
  done < <(printf '%s\n' "${TARGETS[$dataset]}" | tr ',' '\n')
done

echo "Prepared ${run_count} disease-classification runs under ${OUTPUT_ROOT}"
