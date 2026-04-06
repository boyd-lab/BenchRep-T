#!/bin/bash
#SBATCH --job-name=giana_scaling
#SBATCH --output=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/logs/giana_scaling/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/logs/giana_scaling/%x_%A_%a.err

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00    

#SBATCH -A akundaje
#SBATCH -p akundaje

#SBATCH --gres=gpu:1
#SBATCH --array=0-1   # 2 diseases → 2 jobs

# ---- env ----
source ~/.bashrc
conda activate giana

# ---- paths ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
DEPTH_INDICES=${REPO_ROOT}/depth_indices/depth_indices_max75k.json.gz
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/giana_scaling

mkdir -p "$LOGDIR" "$RESULTS"

cd "${REPO_ROOT}"

# ---- diseases ----
DISEASES=("Lupus" "HIV")

DISEASE=${DISEASES[$SLURM_ARRAY_TASK_ID]}

# ---- rename job for nicer logs ----
scontrol update JobId=$SLURM_JOB_ID JobName=giana_scaling_${DISEASE}

echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running scaling experiment for: $DISEASE"
echo "Node: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "======================================="

# ---- run scaling experiment ----
CUDA_VISIBLE_DEVICES=0 python -u -m evals.sequencing_depth_experiment \
  --model giana_2021 \
  --target_disease "$DISEASE" \
  --metadata_path "$METADATA" \
  --repertoire_data_dir "$REPERTOIRE_DIR" \
  --depth_indices "$DEPTH_INDICES" \
  --output_json "${RESULTS}/giana_2021_${DISEASE}_scaling.json"

echo "Done scaling: $DISEASE"
