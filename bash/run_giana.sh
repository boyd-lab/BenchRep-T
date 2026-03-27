REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench
cd ${REPO_ROOT}

CUDA_VISIBLE_DEVICES=0 python -m evals.giana_2020_disease_classification \
    --metadata_path ${REPO_ROOT}/data/malid/metadata.tsv \
    --repertoire_data_dir ${REPO_ROOT}/data/malid/TCR \
    --target_disease Lupus \
    --giana_dir /oak/stanford/groups/akundaje/abuen/tcr-bench/GIANA \
    --output_csv ${REPO_ROOT}/results/giana_2020_lupus_classification.csv
