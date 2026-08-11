# Mal-ID-Lite goes here

This directory is intentionally empty in the repository. Mal-ID-Lite is
released and maintained separately, and this repository is currently anonymized for
review, so it is not vendored here as a git submodule.

## Download it

1. Open the anonymized Mal-ID-Lite repository:
   https://anonymous.4open.science/r/Mal-ID-Lite/README.md
2. Click **"Full repo ZIP"** to download the whole codebase as a zip archive.
3. Extract it. This produces a `Mal-ID-Lite/` folder containing `malid_lite/`,
   `scripts/`, `tests/`, `LICENSE`, `README.md`, `requirements.txt`, and a few
   other top-level docs. Move that folder's *contents* into this directory
   (`models/Mal-ID-Lite/`), replacing this file, so that
   `models/Mal-ID-Lite/malid_lite/`, `models/Mal-ID-Lite/scripts/`, and
   `models/Mal-ID-Lite/README.md` exist directly.

`evals/mal_id_lite_disease_classification.py`, `evals/mal_id_lite_depth_experiment.py`,
and `evals/resource_benchmark.py` check for this content at startup and fail
with a clear error if it is missing.
