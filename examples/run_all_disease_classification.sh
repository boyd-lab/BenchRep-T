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
# DeepRC and DeepTCR were benchmarked at these batch sizes rather than their
# evaluators' argparse defaults (4 and 25), so they are set explicitly here.
DEEPRC_BATCH_SIZE="${DEEPRC_BATCH_SIZE:-32}"
DEEPTCR_BATCH_SIZE="${DEEPTCR_BATCH_SIZE:-4}"
METHODS="${METHODS:-emerson ostmeyer ensemble_regression ensemble_xgboost abmil deeprc deeptcr giana malid_lite}"
DATASETS="${DATASETS:-malid mitchell-t1d rawat-t1d tb ra cmv malid+mitchell-t1d}"

declare -A MODULES=(
  [emerson]=evals.emerson_2017_disease_classification
  [ostmeyer]=evals.ostmeyer_2019_disease_classification
  [ensemble_regression]=evals.ensemble_regression_disease_classification
  [ensemble_xgboost]=evals.ensemble_xgboost_disease_classification
  [abmil]=evals.ensemble_abmil_disease_classification
  [deeprc]=evals.deeprc_2020_disease_classification
  [deeptcr]=evals.deeptcr_2021_disease_classification
  [giana]=evals.giana_2021_disease_classification
  [malid_lite]=evals.mal_id_lite_disease_classification
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
  [malid_lite]=mal_id_lite
)
declare -A TARGETS=(
  [malid]="HIV,T1D,Lupus,Covid19,Influenza"
  [mitchell-t1d]="T1D"
  [rawat-t1d]="T1D"
  [tb]="Progressor"
  [ra]="Rheumatoid Arthritis"
  [cmv]="CMV"
  [malid+mitchell-t1d]="T1D"
)
# Pooled datasets. Rather than being a cohort of their own, these pool a second
# cohort into the same three-fold CV split as the first (utils.cohort_merge),
# which is how the manuscript's united T1D evaluation is run: Zaslavsky/Mal-ID
# specimens and Mitchell specimens are trained and tested together, each
# keeping its own preassigned fold. Every method receives the same --ext_*
# arguments, and each evaluator canonicalizes V/J gene labels across the two
# naming conventions once merging is active. Nothing is copied or symlinked:
# both cohorts are read in place from their own staged directories.
declare -A POOL_INTERNAL=(
  [malid+mitchell-t1d]=malid
)
declare -A POOL_EXTERNAL=(
  [malid+mitchell-t1d]=mitchell-t1d
)
# Staged cohorts use one filename convention regardless of their native one.
POOL_FILE_TEMPLATE='part_table_{participant_label}_{specimen_label}.tsv.gz'

# Expand pooled datasets into the underlying cohorts that must be fetched and
# staged, dropping duplicates (malid is usually requested on its own too).
cohorts_for_datasets() {
  local dataset seen=" "
  for dataset in "$@"; do
    local -a parts=("${dataset}")
    if [[ -n "${POOL_INTERNAL[$dataset]+x}" ]]; then
      parts=("${POOL_INTERNAL[$dataset]}" "${POOL_EXTERNAL[$dataset]}")
    fi
    local part
    for part in "${parts[@]}"; do
      if [[ "${seen}" != *" ${part} "* ]]; then
        seen+="${part} "
        printf '%s\n' "${part}"
      fi
    done
  done
}

python_command() {
  local method=$1
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' "${CONDA_EXE}" run --no-capture-output -n "${ENVIRONMENTS[$method]}" python
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  # Fetch only the cohorts this run needs; the full dataset is 14.6 GiB and a
  # single-cohort run such as DATASETS=ra needs 0.04 GiB of it.
  download=(
    "${PYTHON_BIN}" -m utils.huggingface_data
    --output-dir "${DATA_ROOT}" --task disease
  )
  while IFS= read -r cohort; do
    download+=(--cohort "${cohort}")
  done < <(cohorts_for_datasets ${DATASETS})
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
  while IFS= read -r cohort; do
    stage_args+=(--cohort "${cohort}")
  done < <(cohorts_for_datasets ${DATASETS})
  (cd "${ROOT}" && "${stage_args[@]}")
fi

run_count=0
for dataset in ${DATASETS}; do
  if [[ -z "${TARGETS[$dataset]+x}" ]]; then
    echo "Unknown dataset: ${dataset}" >&2
    exit 2
  fi
  # A pooled dataset reads its primary cohort from the internal cohort's staged
  # directory and merges the external one in at runtime; a plain dataset is its
  # own internal cohort and adds no --ext_* arguments.
  internal="${POOL_INTERNAL[$dataset]:-${dataset}}"
  metadata="${STAGE_ROOT}/${internal}/metadata.tsv"
  repertoires="${STAGE_ROOT}/${internal}/repertoires"
  ext_args=()
  if [[ -n "${POOL_EXTERNAL[$dataset]+x}" ]]; then
    external="${POOL_EXTERNAL[$dataset]}"
    ext_args=(
      --ext_metadata_path "${STAGE_ROOT}/${external}/metadata.tsv"
      --ext_data_dir "${STAGE_ROOT}/${external}/repertoires"
      --ext_file_template "${POOL_FILE_TEMPLATE}"
    )
  fi

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
      command+=(${ext_args[@]+"${ext_args[@]}"})
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
          command+=(
            --results_dir "${method_root}/models" --n_worker_processes "${N_JOBS}"
            --batch_size "${DEEPRC_BATCH_SIZE}"
          )
          [[ "${USE_GPU}" == "1" ]] && command+=(--device cuda:0) || command+=(--device cpu)
          ;;
        deeptcr)
          command+=(
            --results_dir "${method_root}/models" --n_jobs "${N_JOBS}"
            --batch_size "${DEEPTCR_BATCH_SIZE}"
          )
          ;;
        giana)
          command+=(--results_dir "${method_root}/models" --n_threads "${N_THREADS}")
          [[ "${USE_GPU}" == "1" ]] && command+=(--use_gpu)
          ;;
        malid_lite)
          # cache_dir is scoped to ${dataset} (not ${target_tag}), so every
          # target disease sharing this dataset reuses one cache/one set of
          # ESM-2 embeddings instead of rebuilding per target. A pooled dataset
          # is a distinct ${dataset}, which also gives it the separate cache it
          # requires: pooling narrows the cohort to one target and forces
          # amino-acid clone IDs, so its cache cannot be shared with malid's.
          command+=(
            --dataset_name "${dataset}"
            --cache_dir "${OUTPUT_ROOT}/${dataset}/malid_lite_cache"
            --model_save_dir "${method_root}/models"
            --n_jobs "${N_JOBS}"
          )
          [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
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
