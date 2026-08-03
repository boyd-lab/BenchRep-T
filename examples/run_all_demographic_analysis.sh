#!/usr/bin/env bash
set -euo pipefail

# Run both demographic-confounding analyses from the BenchRep-T README.
# Repertoires remain in the downloaded Hugging Face tree; staging creates only
# normalized metadata and symlinks. Run from any directory.
#
#   matched_controls  Replace each disease's random healthy-control pool with a
#                     subset matched on its dominant confounder (age for Lupus,
#                     Influenza and Covid19; African ancestry for HIV), and
#                     compare against five equally sized random draws. This runs
#                     the ordinary disease-classification evaluators with
#                     --adjust_distribution_by_demographics (matched) or
#                     --random_baseline_seeds (control), so each method keeps its
#                     published configuration and only the cohort changes.
#                     7 methods x 4 diseases x 2 modes = 56 runs.
#
#   feature_concat    Concatenate repertoire features with age, sex and ancestry
#                     and compare against the same model on the same
#                     complete-demographics subset, plus a demographics-only
#                     logistic regression. Seven cells per disease, matching the
#                     cells that scripts/democonfound_metrics.py reads.
#                     7 cells x 4 diseases = 28 runs.
#
# GIANA is absent from matched_controls on purpose: its evaluator accepts
# --adjust_distribution_by_demographics but has no --random_baseline_seeds, so a
# matched run would have no paired random-control baseline, and it is omitted
# from the published figure.
#
# malid_lite (Mal-ID-Lite) supports both flags but is not in the default
# MATCHED_METHODS: each (disease, mode) cell -- and each of the 5 baseline
# seeds within a "baseline" cell -- is a full Mal-ID-Lite training run with
# its own cache (the resampled healthy cohort differs per seed, so nothing is
# reused across them), which is expensive to run by default. Opt in with
# MATCHED_METHODS=malid_lite.
#
# DeepRC and DeepTCR here follow the published demographic runs
# (bash/demographic_subsample/), whose training schedules differ from those
# evaluators' argparse defaults; see the DEEPRC_* and DEEPTCR_* knobs below.

ROOT="${BENCHREP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/all_demographic_analysis/${RUN_ID}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}/staged_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA="${USE_CONDA:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_GPU="${USE_GPU:-1}"
N_JOBS="${N_JOBS:-8}"
MAX_FOLDS="${MAX_FOLDS:-}"
ANALYSES="${ANALYSES:-matched_controls feature_concat}"
MATCHED_METHODS="${MATCHED_METHODS:-emerson ostmeyer ensemble_regression ensemble_xgboost abmil deeprc deeptcr}"
MATCHED_DISEASES="${MATCHED_DISEASES:-Lupus HIV Influenza Covid19}"
MATCHED_MODES="${MATCHED_MODES:-adjust baseline}"
BASELINE_SEEDS="${BASELINE_SEEDS:-7 14 21 28 35}"
CONCAT_CELLS="${CONCAT_CELLS:-demographics_only vj_repertoire_only vj_plus_demographics deeptcr_repertoire_only deeptcr_plus_demographics abmil_repertoire_only abmil_plus_demographics}"
CONCAT_DISEASES="${CONCAT_DISEASES:-Lupus HIV Covid19 CMV}"
DEMOGRAPHIC_FEATURES="${DEMOGRAPHIC_FEATURES:-age sex ancestry}"
DUMP_COHORTS="${DUMP_COHORTS:-1}"
DEEPRC_BATCH_SIZE="${DEEPRC_BATCH_SIZE:-32}"
DEEPRC_N_UPDATES="${DEEPRC_N_UPDATES:-10000}"
DEEPRC_EVALUATE_AT="${DEEPRC_EVALUATE_AT:-1000}"
DEEPRC_SAMPLE_N_SEQUENCES="${DEEPRC_SAMPLE_N_SEQUENCES:-10000}"
DEEPTCR_BATCH_SIZE="${DEEPTCR_BATCH_SIZE:-4}"

declare -A MODULES=(
  [emerson]=evals.emerson_2017_disease_classification
  [ostmeyer]=evals.ostmeyer_2019_disease_classification
  [ensemble_regression]=evals.ensemble_regression_disease_classification
  [ensemble_xgboost]=evals.ensemble_xgboost_disease_classification
  [abmil]=evals.ensemble_abmil_disease_classification
  [deeprc]=evals.deeprc_2020_disease_classification
  [deeptcr]=evals.deeptcr_2021_disease_classification
  [malid_lite]=evals.mal_id_lite_disease_classification
  [demographics_only]=evals.demographic_features_disease_classification
  [vj_repertoire_only]=evals.ensemble_regression_disease_classification
  [vj_plus_demographics]=evals.vjgene_demographics_disease_classification
  [deeptcr_repertoire_only]=evals.deeptcr_2021_disease_classification
  [deeptcr_plus_demographics]=evals.deeptcr_demographics_disease_classification
  [abmil_repertoire_only]=evals.ensemble_abmil_disease_classification
  [abmil_plus_demographics]=evals.abmil_demographics_disease_classification
)
declare -A ENVIRONMENTS=(
  [emerson]=benchrep-base
  [ostmeyer]=benchrep-base
  [ensemble_regression]=benchrep-base
  [ensemble_xgboost]=benchrep-base
  [abmil]=abmil
  [deeprc]=deeprc
  [deeptcr]=deeptcr
  [malid_lite]=mal_id_lite
  [demographics_only]=benchrep-base
  [vj_repertoire_only]=benchrep-base
  [vj_plus_demographics]=benchrep-base
  [deeptcr_repertoire_only]=deeptcr
  [deeptcr_plus_demographics]=deeptcr
  [abmil_repertoire_only]=abmil
  [abmil_plus_demographics]=abmil
)
# Diseases with a rule in utils.cohort_adjustments; anything else would run with
# its cohort silently unchanged.
declare -A MATCHED_SUPPORTED=(
  [HIV]=1
  [Lupus]=1
  [T1D]=1
  [Influenza]=1
  [Covid19]=1
)
# Cohort each feature_concat disease is drawn from.
declare -A DISEASE_COHORT=(
  [Lupus]=malid
  [HIV]=malid
  [Influenza]=malid
  [Covid19]=malid
  [CMV]=cmv
)
# Only the base disease-classification evaluators accept --max_folds; the four
# demographics-concatenation evaluators do not.
declare -A SUPPORTS_MAX_FOLDS=(
  [emerson]=1
  [ostmeyer]=1
  [ensemble_regression]=1
  [ensemble_xgboost]=1
  [abmil]=1
  [deeprc]=1
  [deeptcr]=1
  [malid_lite]=1
  [vj_repertoire_only]=1
  [deeptcr_repertoire_only]=1
  [abmil_repertoire_only]=1
)

read -r -a BASELINE_SEED_LIST <<< "${BASELINE_SEEDS}"
read -r -a DEMOGRAPHIC_FEATURE_LIST <<< "${DEMOGRAPHIC_FEATURES}"

has_analysis() { [[ " ${ANALYSES} " == *" $1 "* ]]; }

python_command() {
  local key=$1
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' "${CONDA_EXE}" run --no-capture-output -n "${ENVIRONMENTS[$key]}" python
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

# DeepTCR's evaluators take an integer --device and have no CPU switch, so CPU
# runs have to hide the devices instead.
launcher_prefix() {
  local key=$1
  if [[ "${USE_GPU}" != "1" && "${ENVIRONMENTS[$key]}" == "deeptcr" ]]; then
    printf '%s\n' env CUDA_VISIBLE_DEVICES=
  fi
}

emit() {
  # emit <label> <output_dir> <command...>
  local label=$1 output_dir=$2
  shift 2
  run_count=$((run_count + 1))
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'RUN %03d [%s]:' "${run_count}" "${label}"
    printf ' %q' "$@"
    printf '\n'
  else
    mkdir -p "${output_dir}"
    echo "[$(date --iso-8601=seconds)] ${label}"
    (cd "${ROOT}" && "$@") 2>&1 | tee "${output_dir}/run.log"
  fi
}

for analysis in ${ANALYSES}; do
  case "${analysis}" in
    matched_controls|feature_concat) ;;
    *) echo "Unknown analysis: ${analysis}" >&2; exit 2 ;;
  esac
done

# Resolve the cohorts that have to be downloaded and staged.
declare -A COHORT_SET=()
if has_analysis matched_controls; then
  COHORT_SET[malid]=1
  for disease in ${MATCHED_DISEASES}; do
    if [[ -z "${MATCHED_SUPPORTED[$disease]+x}" ]]; then
      echo "No demographic adjustment defined for disease: ${disease}" >&2
      exit 2
    fi
  done
  for method in ${MATCHED_METHODS}; do
    if [[ -z "${MODULES[$method]+x}" || -z "${SUPPORTS_MAX_FOLDS[$method]+x}" ]]; then
      echo "Unknown matched-controls method: ${method}" >&2
      exit 2
    fi
  done
  for mode in ${MATCHED_MODES}; do
    case "${mode}" in
      adjust|baseline) ;;
      *) echo "Unknown matched-controls mode: ${mode}" >&2; exit 2 ;;
    esac
  done
fi
if has_analysis feature_concat; then
  for disease in ${CONCAT_DISEASES}; do
    if [[ -z "${DISEASE_COHORT[$disease]+x}" ]]; then
      echo "Unknown feature-concat disease: ${disease}" >&2
      exit 2
    fi
    COHORT_SET[${DISEASE_COHORT[$disease]}]=1
  done
  for cell in ${CONCAT_CELLS}; do
    if [[ -z "${MODULES[$cell]+x}" ]]; then
      echo "Unknown feature-concat cell: ${cell}" >&2
      exit 2
    fi
  done
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download=("${PYTHON_BIN}" -m utils.huggingface_data --output-dir "${DATA_ROOT}" --task demographics)
  for cohort in "${!COHORT_SET[@]}"; do
    download+=(--cohort "${cohort}")
  done
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
  for cohort in "${!COHORT_SET[@]}"; do
    stage_args+=(--cohort "${cohort}")
  done
  (cd "${ROOT}" && "${stage_args[@]}")
fi

run_count=0

# ---------------------------------------------------------------------------
# Analysis 1: demographic-matched healthy controls vs random healthy controls
# ---------------------------------------------------------------------------
if has_analysis matched_controls; then
  analysis_root="${OUTPUT_ROOT}/matched_controls"
  metadata="${STAGE_ROOT}/malid/metadata.tsv"
  repertoires="${STAGE_ROOT}/malid/repertoires"

  # Sample lists documenting exactly which specimens each cohort contains.
  if [[ "${DUMP_COHORTS}" == "1" ]]; then
    mapfile -t dump_launcher < <(python_command emerson)
    dump=(
      "${dump_launcher[@]}" -u dump_demographic_cohorts.py
      --metadata_path "${metadata}"
      --out_dir "${analysis_root}/cohort_samples"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'COHORTS:'
      printf ' %q' "${dump[@]}"
      printf '\n'
    else
      mkdir -p "${analysis_root}/cohort_samples"
      echo "[$(date --iso-8601=seconds)] dump matched and random cohort sample lists"
      (cd "${ROOT}" && "${dump[@]}") 2>&1 \
        | tee "${analysis_root}/cohort_samples/dump.log"
    fi
  fi

  for disease in ${MATCHED_DISEASES}; do
    for method in ${MATCHED_METHODS}; do
      for mode in ${MATCHED_MODES}; do
        run_root="${analysis_root}/${disease}/${method}/${mode}"
        mapfile -t launcher < <(python_command "${method}")
        command=()
        mapfile -t prefix < <(launcher_prefix "${method}")
        [[ "${#prefix[@]}" -gt 0 ]] && command+=("${prefix[@]}")
        command+=(
          "${launcher[@]}" -u -m "${MODULES[$method]}"
          --metadata_path "${metadata}"
          --repertoire_data_dir "${repertoires}"
          --target_disease "${disease}"
          --output_csv "${run_root}/scores.csv"
        )

        case "${method}" in
          emerson|ostmeyer)
            ;;
          ensemble_regression)
            command+=(--n_jobs "${N_JOBS}")
            ;;
          ensemble_xgboost)
            command+=(--n_jobs "${N_JOBS}")
            [[ "${USE_GPU}" == "1" ]] && command+=(--xgb_device cuda)
            ;;
          abmil)
            [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
            ;;
          deeprc)
            command+=(
              --results_dir "${run_root}/models"
              --n_worker_processes "${N_JOBS}"
              --batch_size "${DEEPRC_BATCH_SIZE}"
              --n_updates "${DEEPRC_N_UPDATES}"
              --evaluate_at "${DEEPRC_EVALUATE_AT}"
              --sample_n_sequences "${DEEPRC_SAMPLE_N_SEQUENCES}"
            )
            [[ "${USE_GPU}" == "1" ]] && command+=(--device cuda:0) || command+=(--device cpu)
            ;;
          deeptcr)
            command+=(
              --results_dir "${run_root}/models"
              --n_jobs "${N_JOBS}"
              --batch_size "${DEEPTCR_BATCH_SIZE}"
              --device 0
            )
            ;;
          malid_lite)
            # Fresh cache per (disease, method, mode) cell; --random_baseline_seeds
            # further scopes a subdirectory per seed internally (see the mode case
            # below), since the resampled healthy cohort differs per seed.
            command+=(
              --dataset_name "malid_${disease}_${mode}"
              --cache_dir "${run_root}/cache"
              --model_save_dir "${run_root}/models"
              --n_jobs "${N_JOBS}"
            )
            [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
            ;;
        esac

        case "${mode}" in
          adjust)
            command+=(--adjust_distribution_by_demographics)
            ;;
          baseline)
            # Implies --adjust_distribution_by_demographics on the disease side
            # and swaps the healthy side for a random draw per seed.
            command+=(--random_baseline_seeds "${BASELINE_SEED_LIST[@]}")
            ;;
        esac

        [[ -n "${MAX_FOLDS}" ]] && command+=(--max_folds "${MAX_FOLDS}")

        emit "matched_controls / ${disease} / ${method} / ${mode}" \
          "${run_root}" "${command[@]}"
      done
    done
  done
fi

# ---------------------------------------------------------------------------
# Analysis 2: repertoire features with and without concatenated demographics
# ---------------------------------------------------------------------------
if has_analysis feature_concat; then
  analysis_root="${OUTPUT_ROOT}/feature_concat"

  for disease in ${CONCAT_DISEASES}; do
    cohort="${DISEASE_COHORT[$disease]}"
    metadata="${STAGE_ROOT}/${cohort}/metadata.tsv"
    repertoires="${STAGE_ROOT}/${cohort}/repertoires"

    for cell in ${CONCAT_CELLS}; do
      run_root="${analysis_root}/${disease}/${cell}"
      mapfile -t launcher < <(python_command "${cell}")
      command=()
      mapfile -t prefix < <(launcher_prefix "${cell}")
      [[ "${#prefix[@]}" -gt 0 ]] && command+=("${prefix[@]}")
      command+=(
        "${launcher[@]}" -u -m "${MODULES[$cell]}"
        --metadata_path "${metadata}"
        --target_disease "${disease}"
        --output_csv "${run_root}/scores.csv"
      )
      # Every cell except the demographics-only baseline reads repertoires.
      [[ "${cell}" != "demographics_only" ]] \
        && command+=(--repertoire_data_dir "${repertoires}")

      case "${cell}" in
        demographics_only)
          command+=(--features "${DEMOGRAPHIC_FEATURE_LIST[@]}")
          ;;
        vj_repertoire_only)
          # V/J sub-model of Ensemble Regression on the same complete-demographics
          # subset the concatenated model is fit on.
          command+=(--submodel vj_only --require_demographics --n_jobs "${N_JOBS}")
          ;;
        vj_plus_demographics)
          ;;
        deeptcr_repertoire_only)
          command+=(
            --require_demographics
            --results_dir "${run_root}/models"
            --n_jobs "${N_JOBS}"
            --batch_size "${DEEPTCR_BATCH_SIZE}"
            --device 0
          )
          ;;
        deeptcr_plus_demographics)
          command+=(
            --results_dir "${run_root}/models"
            --n_jobs "${N_JOBS}"
            --batch_size "${DEEPTCR_BATCH_SIZE}"
            --device 0
          )
          ;;
        abmil_repertoire_only)
          command+=(--require_demographics)
          [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
          ;;
        abmil_plus_demographics)
          [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)
          ;;
      esac

      if [[ -n "${MAX_FOLDS}" && -n "${SUPPORTS_MAX_FOLDS[$cell]+x}" ]]; then
        command+=(--max_folds "${MAX_FOLDS}")
      fi

      emit "feature_concat / ${disease} / ${cell}" "${run_root}" "${command[@]}"
    done
  done
fi

echo "Prepared ${run_count} demographic-analysis runs under ${OUTPUT_ROOT}"
