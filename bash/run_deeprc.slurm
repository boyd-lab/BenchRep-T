#!/bin/bash
#SBATCH --job-name=deeprc
#SBATCH --output=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/logs/deeprc/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/logs/deeprc/%x_%A_%a.err

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00

#SBATCH -A akundaje
#SBATCH -p akundaje

#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# ---- env ----
source ~/.bashrc
conda activate deeprc

# ---- paths ----
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
METADATA=${REPO_ROOT}/data/malid_clean/metadata.tsv
REPERTOIRE_DIR=${REPO_ROOT}/data/malid_clean/TCR
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/deeprc

mkdir -p "$LOGDIR" "$RESULTS"

cd "${REPO_ROOT}"

# ---- diseases ----
DISEASES=("Lupus" "T1D" "HIV" "Influenza" "Covid19")

DISEASE=${DISEASES[$SLURM_ARRAY_TASK_ID]}

# ---- rename job so %x includes disease ----
scontrol update JobId=$SLURM_JOB_ID JobName=deeprc_${DISEASE}

echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running disease: $DISEASE"
echo "Node: $(hostname)"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "======================================="

# ---- run ----
CUDA_VISIBLE_DEVICES=0 python -u -m evals.deeprc_2020_disease_classification \
  --metadata_path "$METADATA" \
  --repertoire_data_dir "$REPERTOIRE_DIR" \
  --target_disease "$DISEASE" \
  --output_csv "${RESULTS}/deeprc_2020_${DISEASE}_classification.csv" \
  --batch_size 32

echo "Done: $DISEASE"