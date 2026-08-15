#!/usr/bin/env bash
set -euo pipefail

# Run the sequencing-depth scaling task for every method that supports it.
#
# Each job re-runs disease classification at every depth D in the published
# indices file, for every replicate, and writes one JSON holding the per
# (depth, replicate) AUROC/AUPR/balanced-accuracy/F1. Depths, replicate count,
# minimum sequence count and RNG seed all come from that file rather than from
# flags, so the subsampling is exactly the published one: nested across depths,
# so the D sequences at depth D are always a subset of those at any larger depth.
#
# Only the Mal-ID cohort is covered: the published indices are keyed to its
# repertoire identifiers. The scaling figure's own Mal-ID method row is the
# original Zaslavsky classifier, not reproduced here; `malid_lite` (this
# repository's Mal-ID-Lite reimplementation) is a separate, runnable stand-in
# for it, but it doesn't go through evals.sequencing_depth_experiment's
# --model dispatch since Mal-ID-Lite is a multi-stage subprocess pipeline
# rather than an in-process evaluator class. Instead, evals.mal_id_lite_depth_experiment
# filters each repertoire file down to the same pre-computed row indices the
# other methods use for a given (depth, repeat), then runs the normal
# Mal-ID-Lite pipeline through it. It is not included in METHODS by default
# (like giana_2021 and ostmeyer_2019) since each (depth, repeat) is a full
# Mal-ID-Lite training run -- opt in explicitly with METHODS=malid_lite.
#
# malid_lite is also structurally different from every other method here: it
# runs ONCE per (depth, repeat), covering every disease in DISEASES together
# (see evals/mal_id_lite_depth_experiment.py's own module docstring), rather
# than once per (disease, depth, repeat) the way the other methods do. This
# matters for correctness, not just speed: Mal-ID-Lite recomputes clone_id
# from scratch on the actual downsampled sequences for each (depth, repeat)
# either way, but every disease is trained against the same healthy/reference
# participants, so sharing that work (and the ESM-2 embeddings built on top of
# it) across every requested disease means that shared reference population is
# only computed once per (depth, repeat), not once per disease -- the
# disease-specific participants themselves are never shared between diseases,
# only the reference class is. Its own output JSON covers
# every (depth, repeat, disease) combination in one file, with a "disease"
# field per result entry, rather than one file per disease the way the other
# methods produce.
#
# Repertoires remain in the downloaded Hugging Face tree; staging creates only
# normalized metadata and symlinks. Run from any directory.
#
# The full matrix is expensive: 6 depths x 5 replicates is 30 complete
# cross-validation runs per (method, disease). DEPTHS and REPLICATES narrow that
# without touching the downloaded indices file, which is what makes a smoke test
# or a split across nodes practical:
#
#   DEPTHS=1000 REPLICATES=1 bash examples/run_all_depth_scaling.sh   # quick check
#   DEPTHS=75000 METHODS=ensemble_xgboost bash examples/run_all_depth_scaling.sh
#
# Both write a filtered copy of the indices under the run directory and tag the
# output filename, so narrowed runs never collide with a full one.

ROOT="${BENCHREP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/all_depth_scaling/${RUN_ID}}"
STAGE_ROOT="${STAGE_ROOT:-${OUTPUT_ROOT}/staged_data}"
DEPTH_INDICES="${DEPTH_INDICES:-${DATA_ROOT}/Mal-ID/scaling_exp_depth_indices_max75k.json.gz}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_EXE="${CONDA_EXE:-conda}"
USE_CONDA="${USE_CONDA:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_GPU="${USE_GPU:-1}"
N_JOBS="${N_JOBS:-8}"
N_THREADS="${N_THREADS:-10}"
RANDOM_SEED="${RANDOM_SEED:-7}"
DEPTHS="${DEPTHS:-}"
REPLICATES="${REPLICATES:-}"
# The published scaling runs kept XGBoost and GIANA on CPU even where a GPU was
# available; USE_GPU=0 forces both regardless.
XGBOOST_DEVICE="${XGBOOST_DEVICE:-cpu}"
GIANA_USE_GPU="${GIANA_USE_GPU:-0}"
METHODS="${METHODS:-emerson_2017 ensemble_regression ensemble_xgboost deeprc_2020}"
DISEASES="${DISEASES:-Lupus HIV}"

COHORT=malid

# --model values accepted by evals.sequencing_depth_experiment, mapped to the
# environment each one needs. The _kmer/_vj entries are sub-model ablations of
# the two ensemble methods.
declare -A ENVIRONMENTS=(
  [emerson_2017]=benchrep-base
  [ostmeyer_2019]=benchrep-base
  [ensemble_regression]=benchrep-base
  [ensemble_regression_kmer]=benchrep-base
  [ensemble_regression_vj]=benchrep-base
  [ensemble_xgboost]=benchrep-base
  [ensemble_xgboost_kmer]=benchrep-base
  [ensemble_xgboost_vj]=benchrep-base
  [giana_2021]=giana
  [deeprc_2020]=deeprc
  [malid_lite]=mal_id_lite
)
# Diseases the published scaling figure covers.
declare -A SUPPORTED_DISEASES=(
  [Lupus]=1
  [HIV]=1
)

python_command() {
  local model=$1
  if [[ "${USE_CONDA}" == "1" ]]; then
    printf '%s\n' "${CONDA_EXE}" run --no-capture-output -n "${ENVIRONMENTS[$model]}" python
  else
    printf '%s\n' "${PYTHON_BIN}"
  fi
}

for disease in ${DISEASES}; do
  if [[ -z "${SUPPORTED_DISEASES[$disease]+x}" ]]; then
    echo "Unknown depth-scaling disease: ${disease}" >&2
    exit 2
  fi
done
for model in ${METHODS}; do
  if [[ -z "${ENVIRONMENTS[$model]+x}" ]]; then
    echo "Unknown depth-scaling model: ${model}" >&2
    exit 2
  fi
done

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download=(
    "${PYTHON_BIN}" -m utils.huggingface_data
    --output-dir "${DATA_ROOT}" --task depth
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

# Narrow the published indices to a depth subset and/or fewer replicates. The
# per-repertoire index arrays are untouched, so the retained depths and
# replicates are bit-identical to a full run's.
indices="${DEPTH_INDICES}"
output_suffix=""
if [[ -n "${DEPTHS}" || -n "${REPLICATES}" ]]; then
  [[ -n "${DEPTHS}" ]] && output_suffix+="_depths${DEPTHS//,/_}"
  [[ -n "${REPLICATES}" ]] && output_suffix+="_reps${REPLICATES}"
  indices="${OUTPUT_ROOT}/depth_indices${output_suffix}.json.gz"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: would filter ${DEPTH_INDICES} -> ${indices}" \
      "(depths=${DEPTHS:-all}, replicates=${REPLICATES:-all})"
  else
    (cd "${ROOT}" && "${PYTHON_BIN}" - \
      "${DEPTH_INDICES}" "${indices}" "${DEPTHS}" "${REPLICATES}" <<'PY'
import gzip
import json
import sys

src, dst, depths_arg, reps_arg = sys.argv[1:5]


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


data = load(src)

if depths_arg:
    keep = [int(value) for value in depths_arg.split(",")]
    available = set(data.get("depths", []))
    missing = [depth for depth in keep if depth not in available]
    if missing:
        raise SystemExit(
            f"Requested depths not present in {src}: {missing}; "
            f"available: {sorted(available)}"
        )
    data["depths"] = keep

if reps_arg:
    n_reps = int(reps_arg)
    if n_reps < 1 or n_reps > data["n_reps"]:
        raise SystemExit(
            f"REPLICATES must be between 1 and {data['n_reps']}, got {n_reps}"
        )
    data["n_reps"] = n_reps

with gzip.open(dst, "wt", encoding="utf-8") as handle:
    json.dump(data, handle)

print(
    f"Filtered depth indices -> {dst} "
    f"(depths={data['depths']}, n_reps={data['n_reps']}, "
    f"repertoires={len(data['repertoires'])})"
)
PY
    )
  fi
fi

run_count=0

# malid_lite: one call covers every disease in DISEASES together, per
# (depth, repeat) -- see the header comment above and
# evals/mal_id_lite_depth_experiment.py's own module docstring for why
# (shares the expensive cache/embeddings build across diseases instead of
# rebuilding it once per disease). Handled entirely separately from the
# per-(disease, model) loop below, which every other method still uses
# unchanged.
for model in ${METHODS}; do
  if [[ "${model}" != "malid_lite" ]]; then
    continue
  fi
  run_root="${OUTPUT_ROOT}/${model}"
  mapfile -t launcher < <(python_command "${model}")
  command=(
    "${launcher[@]}" -u -m evals.mal_id_lite_depth_experiment
    --target_diseases ${DISEASES}
    --metadata_path "${metadata}"
    --repertoire_data_dir "${repertoires}"
    --depth_indices "${indices}"
    --dataset_name "${COHORT}"
    --cache_root "${run_root}/cache"
    --n_jobs "${N_JOBS}"
    --output_json "${run_root}/scaling${output_suffix}.json"
  )
  [[ "${USE_GPU}" != "1" ]] && command+=(--no_gpu)

  run_count=$((run_count + 1))
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'RUN %03d:' "${run_count}"
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    mkdir -p "${run_root}"
    echo "[$(date --iso-8601=seconds)] ${COHORT} / diseases=${DISEASES} / ${model}" \
      "(depths=${DEPTHS:-all}, replicates=${REPLICATES:-all})"
    (cd "${ROOT}" && "${command[@]}") 2>&1 | tee "${run_root}/run${output_suffix}.log"
  fi
done

# Every other method: one run per (disease, model) -- unchanged.
for disease in ${DISEASES}; do
  for model in ${METHODS}; do
    if [[ "${model}" == "malid_lite" ]]; then
      continue  # handled above, once, covering every disease together
    fi
    run_root="${OUTPUT_ROOT}/${disease}/${model}"
    mapfile -t launcher < <(python_command "${model}")

    command=()
    if [[ "${model}" == "deeprc_2020" && "${USE_GPU}" != "1" ]]; then
      # DeepRC is constructed without a device argument here, so CPU-only runs
      # have to hide the GPUs.
      command+=(env CUDA_VISIBLE_DEVICES=)
    fi

    command+=(
      "${launcher[@]}" -u -m evals.sequencing_depth_experiment
      --model "${model}"
      --target_disease "${disease}"
      --metadata_path "${metadata}"
      --repertoire_data_dir "${repertoires}"
      --depth_indices "${indices}"
      --random_seed "${RANDOM_SEED}"
      --output_json "${run_root}/scaling${output_suffix}.json"
    )

    case "${model}" in
      ensemble_xgboost*)
        device="${XGBOOST_DEVICE}"
        [[ "${USE_GPU}" != "1" ]] && device=cpu
        command+=(--xgboost_n_jobs "${N_JOBS}" --xgboost_device "${device}")
        ;;
      giana_2021)
        # GIANA re-clusters at every depth and replicate, so each one gets its
        # own directory under the run; the evaluator appends a depth/replicate
        # tag beneath this path.
        command+=(
          --giana_results_dir "${run_root}/clusters"
          --giana_n_threads "${N_THREADS}"
          --giana_threshold_iso 7
        )
        if [[ "${GIANA_USE_GPU}" != "1" || "${USE_GPU}" != "1" ]]; then
          command+=(--giana_cpu)
        fi
        ;;
    esac

    run_count=$((run_count + 1))
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'RUN %03d:' "${run_count}"
      printf ' %q' "${command[@]}"
      printf '\n'
    else
      mkdir -p "${run_root}"
      echo "[$(date --iso-8601=seconds)] ${COHORT} / ${disease} / ${model}" \
        "(depths=${DEPTHS:-all}, replicates=${REPLICATES:-all})"
      (cd "${ROOT}" && "${command[@]}") 2>&1 | tee "${run_root}/run${output_suffix}.log"
    fi
  done
done

echo "Prepared ${run_count} depth-scaling runs under ${OUTPUT_ROOT}"
