# Bash Script Layout

Scripts are grouped by experiment/workflow:

- `base/`: main per-method disease classification runs.
- `scaling/`: scaling/depth experiments and scheduler wrappers.
- `demographic_subsample/`: demographic-matched subsampling, random baselines, and demographic-complete subset controls.
- `predict_with_demographics/`: models that explicitly use demographic covariates as predictors.
- `external/`: external-cohort experiments.
- `drivers/`: driver/public-clone identification experiments.
- `preprocessing/`: one-off data preprocessing helpers.

Most scripts assume `REPO_ROOT=/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench`
and can be launched from anywhere.
