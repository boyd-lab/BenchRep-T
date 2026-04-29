#!/bin/bash
set -uo pipefail

# ---- flags ----
DEBUG=false
N_JOBS=10

for arg in "$@"; do
  case $arg in
    --debug) DEBUG=true ;;
    --n_jobs=*) N_JOBS="${arg#*=}" ;;
  esac
done

# ---- config ----
# Only Tb external data is used — no MAL-ID samples.
# Controller is treated as Healthy/Background; Progressor is the disease class.
# The harmonized metadata is passed as both --metadata_path and --ext_metadata_path.
# When processed as internal, add_file_paths constructs part_table_*_*.tsv.gz paths
# that don't exist in data_cleaned/, so filter_existing_files drops all rows.
# prepare_merged_cohort then re-adds those samples using the correct file template.
REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
MODEL_SAVE_DIR=${REPO_ROOT}/results/ensemble_xgboost_models_ext
RESULTS=${REPO_ROOT}/results
LOGDIR=${REPO_ROOT}/logs/ensemble_xgboost_ext

EXT_RAW_METADATA=${REPO_ROOT}/data/external_airr/tuberculosis/metadata_Tb.tsv
EXT_METADATA=${REPO_ROOT}/data/external_airr/tuberculosis/metadata_Tb_harmonized.tsv
EXT_DATA_DIR=${REPO_ROOT}/data/external_airr/tuberculosis/data_cleaned
EXT_FILE_TEMPLATE='{participant_label}.tsv'

TARGET_DISEASE="Tuberculosis"

mkdir -p "$LOGDIR" "$RESULTS" "$MODEL_SAVE_DIR"

# Harmonize Tb metadata to MAL-ID column style if not already done.
# Controller -> Healthy/Background, Progressor -> Tuberculosis.
if [ ! -f "$EXT_METADATA" ]; then
  echo "Harmonizing Tb metadata to MAL-ID format -> $EXT_METADATA"
  python -c "
import sys, pandas as pd
raw, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(raw, sep='\t')
harmonized = pd.DataFrame({
    'participant_label': df['sample_name'],
    'specimen_label':    df['sample_name'],
    'disease':           df['Group'].map({
                             'Controller': 'Healthy/Background',
                             'Progressor': 'Tuberculosis',
                         }),
    'specimen_time_point': df['Study Day'].fillna('').astype(str),
    'study_name':        'external_Tb',
    'available_gene_loci': 'GeneLocus.TCR',
    'malid_cross_validation_fold_id_when_in_test_set': df['fold'],
})
if harmonized['disease'].isna().any():
    bad = df.loc[harmonized['disease'].isna(), 'Group'].unique()
    sys.exit(f'Unmapped Group values: {bad}')
harmonized.to_csv(out, sep='\t', index=False)
print(f'Wrote {len(harmonized)} rows to {out}')
print(harmonized['disease'].value_counts().to_string())
" "$EXT_RAW_METADATA" "$EXT_METADATA"
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
log="${LOGDIR}/ensemble_xgboost_ext_Tb_${RUN_TS}.log"
echo "[$(date +%T)] start Tb ext classification -> $log"

{
  echo "[$(date +%T)] start Tb ext classification"

  extra_flags=()
  if $DEBUG; then
    extra_flags+=(--debug_repertoires 10)
  fi

  cd "${REPO_ROOT}"
  python -u -m evals.ensemble_xgboost_disease_classification \
    --metadata_path "$EXT_METADATA" \
    --repertoire_data_dir "$EXT_DATA_DIR" \
    --target_disease "$TARGET_DISEASE" \
    --n_jobs "$N_JOBS" \
    --ext_metadata_path "$EXT_METADATA" \
    --ext_data_dir "$EXT_DATA_DIR" \
    --ext_file_template "$EXT_FILE_TEMPLATE" \
    --output_csv "${RESULTS}/ensemble_xgboost_ext_Tb_classification.csv" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    "${extra_flags[@]}"

  status=$?
  echo "[$(date +%T)] done Tb ext classification (exit $status)"
  exit $status
} >"$log" 2>&1

echo "[$(date +%T)] done Tb | log: $log"
