# BenchRep-T: A Systematic Evaluation of T-Cell Repertoire-Based Disease Diagnostics

A unified benchmark for TCRβ repertoire-based disease classification,
harmonizing cohorts from Zaslavsky et al. (Mal-ID), Mitchell, Rawat, Musvosvi,
Savola, and Emerson into a single AIRR-compliant schema. BenchRep-T evaluates
statistical, feature-engineered, clustering, and deep-learning methods using
identical inputs and fixed splits.

## Overview

BenchRep-T includes the following disease and serostatus tasks:

- **HIV, Lupus, Influenza, and COVID-19** in the Zaslavsky/Mal-ID cohort
- **T1D** across the Zaslavsky/Mal-ID, Mitchell, and Rawat cohorts, supporting
  both within-cohort and cross-cohort evaluation
- **Tuberculosis progression** in the Musvosvi cohort
- **Rheumatoid Arthritis** in the Savola cohort
- **CMV serostatus** in the Emerson cohort

It defines four evaluation tasks: disease classification, driver-sequence identification, sequencing-depth scaling, and demographic-confounding analysis. All methods consume identical AIRR-formatted repertoire files and are scored on pre-assigned 3-fold cross-validation splits.

## Setup

Requires Python >= 3.10. The recommended installation uses one minimal Conda
environment per method family so that DeepTCR's TensorFlow constraints do
not conflict with the newer PyTorch and scientific-Python stacks. ABMIL, GIANA,
DeepRC, DeepTCR, and Mal-ID-Lite are separated from `benchrep-base` to make
dependency resolution easier and keep each method's framework and version
constraints isolated.

Mal-ID-Lite is developed and released separately, and this repository is currently
anonymized for review, so it is not vendored here as a git
submodule. Download it manually and place it at `models/Mal-ID-Lite`:

1. Open the anonymized Mal-ID-Lite repository:
   https://anonymous.4open.science/r/Mal-ID-Lite/README.md
2. Click **"Full repo ZIP"** to download the whole codebase as a zip archive.
3. Extract it. This produces a `Mal-ID-Lite/` folder containing `malid_lite/`,
   `scripts/`, `tests/`, `LICENSE`, `README.md`, `requirements.txt`, and a few
   other top-level docs. Move that folder's *contents* into this repository's
   `models/Mal-ID-Lite/` directory, so that `models/Mal-ID-Lite/malid_lite/`,
   `models/Mal-ID-Lite/scripts/`, and `models/Mal-ID-Lite/README.md` exist
   directly.

`evals/mal_id_lite_disease_classification.py`, `evals/mal_id_lite_depth_experiment.py`,
and `evals/resource_benchmark.py` check for this at startup and fail with a
clear error if it is missing.

| Environment | File | Methods |
|-------------|------|---------|
| `benchrep-base` | `environment-base.yml` | Emerson, Ostmeyer, and Ensemble Regression/XGBoost |
| `abmil` | `environment-abmil.yml` | ABMIL and pretrained-encoder ABMIL variants |
| `giana` | `environment-giana.yml` | GIANA |
| `deeprc` | `environment-deeprc.yml` | DeepRC |
| `deeptcr` | `environment-deeptcr.yml` | DeepTCR |
| `mal_id_lite` | `environment-mal-id-lite.yml` | Mal-ID-Lite |

The environments are separated as follows:

- **`benchrep-base`** uses Python 3.11 and contains the shared scientific
  stack and XGBoost. It runs Emerson, Ostmeyer, Ensemble Regression, and
  Ensemble XGBoost, as well as the common preprocessing and analysis
  utilities. The project prefix distinguishes it from Conda's special `base`
  environment.
- **`abmil`** uses Python 3.10 with PyTorch and Transformers. It runs standard
  ABMIL and the pretrained TCR-encoder variants without pulling TensorFlow into
  the environment.
- **`giana`** uses Python 3.11 with Biopython and CUDA 12 FAISS. The package
  implements both CPU indexing and the GPU API used by `USE_GPU=1`; a compatible
  NVIDIA driver is only required for GPU runs. Keeping GIANA separate avoids
  coupling its FAISS and NumPy constraints to the other environments.
- **`deeprc`** uses Python 3.10 with PyTorch, HDF5, TensorBoard, the upstream
  `deeprc` package, and `widis-lstm-tools`. The PyTorch wheel supplies the CUDA
  runtime used by the GPU launchers.
- **`deeptcr`** uses Python 3.10 and the legacy-compatible TensorFlow 2.15,
  NumPy 1.23, pandas 1.5, scikit-learn 1.2, and Biopython 1.76 stack required by
  the bundled DeepTCR TensorFlow-1 compatibility and deprecated Biopython APIs.
  It is intentionally isolated from the newer scientific-Python environments.
- **`mal_id_lite`** uses Python 3.12 with PyTorch, `fair-esm` (for Model 3's
  ESM-2 embeddings), and `python-glmnet` (for Model 1's elastic-net logistic
  regression). It runs Mal-ID-Lite, downloaded separately into
  `models/Mal-ID-Lite` (see Setup above).

Create the environments from the repository root:

```bash
conda env create -f environment-base.yml
conda env create -f environment-abmil.yml
conda env create -f environment-giana.yml
conda env create -f environment-deeprc.yml
conda env create -f environment-deeptcr.yml
conda env create -f environment-mal-id-lite.yml
```

Install BenchRep-T itself into each environment in editable mode. Dependencies
are already supplied by the corresponding YAML, so `--no-deps` prevents pip
from changing the tested method-specific pins:

```bash
for env_name in benchrep-base abmil giana deeprc deeptcr mal_id_lite; do
    conda run -n "${env_name}" python -m pip install --no-deps -e .
done
```

For GPU execution, PyTorch includes its CUDA runtime dependencies. DeepTCR uses
TensorFlow's `and-cuda` extra, which supplies a matched CUDA 12 runtime, cuDNN,
and PTX compiler while retaining the TensorFlow 1 compatibility APIs used by
the bundled source, so it does not require a system CUDA toolkit. GIANA
similarly uses a self-contained CUDA 12 FAISS wheel. These GPU execution paths
were validated on an H200 and an L40S.

Because each environment ships its own CUDA runtime, a system CUDA on
`LD_LIBRARY_PATH` can shadow those wheels and break an otherwise correct
environment. With `/usr/local/cuda-12.0/lib64` exported, for example, importing
PyTorch in the `abmil` environment fails with:

```text
ImportError: .../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol:
__nvJitLinkAddData_12_1, version libnvJitLink.so.12
```

CUDA 12.0's `libnvJitLink.so.12` is loaded ahead of the wheel's and does not
export the symbol that the bundled cuSPARSE 12.1 needs. The environments
themselves are correct; unset or reorder `LD_LIBRARY_PATH` so the wheel-provided
libraries win:

```bash
env -u LD_LIBRARY_PATH bash examples/run_all_disease_classification.sh
```

Confirm an environment resolves its own CUDA libraries with:

```bash
conda run -n abmil python -c "import torch; print(torch.cuda.is_available())"
```

### uv or pip

The same dependency groups are exposed as optional extras in `pyproject.toml`.
Install with [uv](https://docs.astral.sh/uv/):

```bash
# Core dependencies
uv sync

# With specific model extras
uv sync --extra base
uv sync --extra abmil
uv sync --extra giana
uv sync --extra deeprc
uv sync --extra deeptcr
uv sync --extra mal-id-lite
uv sync --extra drivers
```

Alternatively, with pip:

```bash
pip install -e .             # core only
pip install -e ".[base]"     # one method-family extra per environment
```

Do not combine all extras in one environment. In particular, DeepTCR requires
Biopython 1.76 while the other method environments use newer Biopython
versions. Use the separate environments described above, or substitute the
desired method-family extra in the pip example.

## Repository Structure

```
models/                          Classification methods
├── emerson_2017.py              Emerson et al. 2017          (statistical)
├── ostmeyer_2019.py             Ostmeyer et al. 2019         (statistical)
├── ensemble_regression.py       V/J + gkmer LogReg           (engineered features)
├── ensemble_xgboost.py          V/J + gkmer XGBoost          (engineered features)
├── GIANA/                       Zhang et al. 2021 (GIANA 4.1) (similarity/clustering)
├── ensemble_abmil.py            ABMIL                        (deep learning)
├── DeepRC/                      Widrich et al. 2020          (deep learning)
├── DeepTCR/                     Sidhom et al. 2021           (deep learning)
└── Mal-ID-Lite/                 Zaslavsky et al. 2025 (reimpl.) (deep learning, downloaded separately)
evals/                           Per-method experiment scripts (disease, drivers, depth, demographics)
examples/                        One runner per evaluation task (see below)
preprocessing/                   Repertoire cleaning and preparation
external_data_process/           Cohort-specific conversion to AIRR and gene harmonization
utils/                           Repertoire I/O, metric helpers, cohort/covariate adjustment
```

## Implemented Methods

### Disease-signature / statistical

- **Emerson et al. 2017** — Identifies disease-associated CDR3 sequences via Fisher's exact test, then scores repertoires using a Beta-Binomial generative model over the discovered sequences.
- **Ostmeyer et al. 2019** — Multiple-instance learning over 4-mer motifs extracted from CDR3 sequences, encoded with Atchley factors and classified by logistic regression with random restarts under a max-aggregation MIL objective.

### Engineered repertoire-level features

- **V/J + gapped-kmer (LogReg)** — Each repertoire is summarized as two feature dictionaries: V/J gene usage and gapped 4-mer frequencies. An L1-penalized logistic regression is trained on each, and the two predictions are linearly combined with a tuned weight α.
- **V/J + gapped-kmer (XGBoost)** — Same feature decomposition as above, with each base learner replaced by a gradient-boosted tree classifier tuned via two-stage grid search with early stopping.

### Similarity / clustering-based

- **GIANA (Zhang et al. 2021)** — Encodes each CDR3 into a 96-dim isometric vector so that Euclidean distance approximates a BLOSUM-weighted sequence distance, then jointly clusters all training and test sequences. Test repertoires are scored by the mean disease fraction of the training-derived clusters their sequences fall into.

### Deep learning

- **ABMIL** — Learned amino-acid and V/J gene embeddings fed through a 1D-CNN encoder, with a gated-attention aggregator pooling per-sequence features into a repertoire-level representation for end-to-end classification.
- **DeepRC (Widrich et al. 2020)** — 1D-CNN sequence embeddings aggregated via a modern Hopfield attention block over up to ~10⁵ sequences per repertoire.
- **DeepTCR (Sidhom et al. 2021)** — Convolutional encoder over CDR3 plus V/D/J gene identities, with attention pooling over a fixed concept bank for repertoire-level prediction (whole-file workflow).
- **Mal-ID-Lite (reimplementation of Zaslavsky et al. 2025)** — Three complementary base models — V-J gene usage (elastic-net logistic regression), convergent CDR3 clusters (hierarchical clustering + Fisher's exact test + GLM), and ESM-2 sequence embeddings (per-V-gene ridge classifiers) — combined by a ridge-regression metamodel. Downloaded separately into `models/Mal-ID-Lite` (see Setup); `evals/mal_id_lite_disease_classification.py` adapts the benchmark's staged metadata/repertoire format to Mal-ID-Lite's own input conventions and converts its output back to the standard scores schema.

## Tasks

### 1. Disease Classification

Binary disease-vs-control classification uses fixed three-fold
cross-validation, with an internal 80/20 or 90/10 train/validation split for
hyperparameter tuning. The following cohort protocols reuse the same evaluation
harness:

- **Zaslavsky/Mal-ID** — HIV, Lupus, Influenza, COVID-19, and T1D are scored
  against Healthy/Background controls using three preassigned folds.
- **Mitchell and Rawat T1D** — Each cohort supports within-cohort evaluation.
  T1D models can also be trained on one cohort and evaluated on the others to
  measure cross-cohort generalization. Zaslavsky/Mal-ID and Mitchell T1D
  repertoires can additionally be pooled for three-fold evaluation, as the
  `malid+mitchell-t1d` dataset (see [Pooled cohorts](#pooled-cohorts)).
- **Musvosvi TB and Savola RA** — Each cohort uses the same three-fold
  within-cohort protocol.
- **Emerson CMV** — CMV-positive versus CMV-negative classification uses fixed
  three-fold splits.

Metrics: AUROC, AUPR.

### 2. Driver Sequence Identification

Tests whether per-sequence scores produced as a by-product of classification
surface known antigen-specific TCRs. Ground truth is built from VDJdb
(confidence ≥ 2), augmented for COVID-19 with experimentally validated
SARS-CoV-2 clonotypes from Minervina et al., and matched to benchmark
repertoires via ≥90% Levenshtein similarity. Supported for Emerson 2017, GIANA,
Ensemble Regression, Ensemble XGBoost, ABMIL, and DeepRC; each ranks sequences
by its native score (Fisher p-value, cluster disease fraction,
decision-function weight, attention weight, etc.). A random-chance baseline is
provided by `compute_random_baseline_recall.py`.

Metrics: recall@k (k ∈ {100, 1000, 10000}, macro-averaged across disease-positive repertoires).

### 3. Sequencing Depth Scaling

**Zaslavsky/Mal-ID cohort only** — reproduces that cohort's published scaling figure; no other cohort has depth indices. Re-runs disease classification after downsampling every repertoire (training and test) to a common target depth D ∈ {1k, 5k, 10k, 25k, 50k, 75k}. Subsampling indices are stored in a published file (`scaling_exp_depth_indices_max75k.json.gz`, generated with a fixed seed) fetched alongside the rest of the dataset from Hugging Face and only read at run time, so every user subsamples the exact same rows. It is **nested** across depths, so the D sequences at depth D are always a subset of those used at any larger depth — differences across depths cannot be explained by drawing different sequences. Five independent replicates per depth; specimens with fewer than 75k unique sequences are excluded entirely (all depths, not just 75k), so every depth in the sweep is evaluated on the exact same population of specimens.

### 4. Demographic Confounding Analysis

Two complementary analyses probe whether classifier signal is carried by participant demographics rather than disease biology:

1. **Demographic-matched controls** — For each disease, the random pool of healthy controls is replaced with a subset matched to the disease cohort's dominant confounder (age for Lupus, Influenza, COVID-19; African ancestry for HIV). Matched-control AUROC is compared against a random-control baseline averaged over five draws of the same size. Sample lists are dumped by `dump_demographic_cohorts.py`.
2. **Demographic-feature concatenation** — Each method's repertoire-level features are concatenated with age, sex, and ancestry; the concatenated model is compared against the base model and a demographics-only logistic-regression baseline. Implemented for Ensemble Regression/XGBoost, ABMIL, and DeepTCR (`*_demographics_disease_classification.py`); standalone demographics-only and V/J-gene-only baselines are also provided.

## Data

### Sources

| Cohort | Prediction task | Labeled repertoires | Class composition |
|--------|-----------------|--------------------:|-------------------|
| Zaslavsky/Mal-ID | HIV, T1D, Lupus, COVID-19, and Influenza vs Healthy/Background | 550 | 197 Healthy/Background; 98 HIV; 96 T1D; 64 Lupus; 58 COVID-19; 37 Influenza |
| Mitchell | T1D vs Healthy/Background | 196 | 171 T1D; 25 Healthy/Background |
| Rawat | T1D vs Healthy/Background | 614 | 426 T1D; 188 Healthy/Background |
| Musvosvi | TB progression vs control | 140 | 63 Progressor; 77 Controller |
| Savola | Rheumatoid Arthritis vs Healthy | 91 | 71 Rheumatoid Arthritis; 20 Healthy |
| Emerson | CMV seropositive vs seronegative | 761 | 340 CMV-positive; 421 CMV-negative |

Demographic annotations are used where they are available.

### Format

Repertoire files are gzip-compressed TSV (`.tsv.gz`) in AIRR standard format. The key columns used by all methods are:

| Column | Description |
|--------|-------------|
| `cdr3_aa` | CDR3 amino acid sequence |
| `v_call` | V gene (IMGT nomenclature, e.g., TRBV7-2\*01) |
| `j_call` | J gene |
| `duplicate_count` | Clone count (optional; assumed 1 if absent) |

Repertoire filenames follow the cohort-native conventions:

| Cohort | Filename template |
|--------|-------------------|
| Zaslavsky/Mal-ID | `part_table_{participant_label}_{specimen_label}.tsv.gz` |
| Mitchell | `{specimen_label}.tsv.gz` |
| Rawat | `{participant_label}_TCRB.tsv.gz` |
| Musvosvi, Savola, Emerson | `{specimen_label}.tsv.gz` |

### Metadata

| Cohort | Metadata file |
|--------|---------------|
| Zaslavsky/Mal-ID | `data/Mal-ID/metadata.tsv` |
| Mitchell | `data/immunoSEQ/Mitchell_T1D/metadata.tsv` |
| Rawat | `data/immunoSEQ/Rawat_T1D/metadata.tsv` |
| Musvosvi | `data/immunoSEQ/Musvosvi_TB/metadata.tsv` |
| Savola | `data/immunoSEQ/Savola_RA/metadata.tsv` |
| Emerson | `data/immunoSEQ/Emerson_CMV/metadata.tsv` |

The harmonized metadata tables use the following common columns:

| Column | Description |
|--------|-------------|
| `participant_label` | Participant identifier |
| `specimen_label` | Specimen identifier |
| `disease` | Disease label (e.g., HIV, Covid19, Healthy/Background) |
| `malid_cross_validation_fold_id_when_in_test_set` or `CV_fold` | Pre-assigned CV fold (0, 1, or 2), depending on cohort |
| `age`, `sex`, `ancestry` | Demographics, where available |

### HuggingFace

Preprocessed BenchRep-T repertoires and metadata are hosted on Hugging Face at:

**[https://huggingface.co/datasets/neurips-2026-dataset/BenchRep-T](https://huggingface.co/datasets/neurips-2026-dataset/BenchRep-T)**

The downloaded dataset has this model-ready structure:

```text
data/
├── Mal-ID/
│   ├── metadata.tsv
│   ├── repertoires/                              # 550 .tsv.gz
│   ├── scaling_exp_depth_indices_max75k.json.gz
│   └── vdjdb_minervina_driver_seq_matches.csv
└── immunoSEQ/
    ├── Savola_RA/{metadata.tsv,repertoires/}      # 91 repertoires
    ├── Musvosvi_TB/{metadata.tsv,repertoires/}    # 140 repertoires
    ├── Rawat_T1D/{metadata.tsv,repertoires/}      # 614 repertoires
    ├── Mitchell_T1D/{metadata.tsv,repertoires/}   # 196 repertoires
    └── Emerson_CMV/{metadata.tsv,repertoires/}    # 761 repertoires
```

BenchRep-T is public. No Hugging Face account, access request, login, or token
is required to download it.

After installing BenchRep-T, materialize the complete dataset in its native
Hugging Face directory layout:

```bash
benchrep-download-data
```

Downloads can be limited by experiment task and cohort. Options are repeatable,
which makes it possible to stage only the data required by a particular job:

```bash
# COVID-19 disease classification and demographic experiments
benchrep-download-data --task disease --task demographics --cohort malid

# Driver-sequence experiment (Mal-ID repertoires plus driver matches)
benchrep-download-data --task drivers

# Sequencing-depth experiment (Mal-ID repertoires plus nested depth indices)
benchrep-download-data --task depth

# Within-cohort CMV disease classification
benchrep-download-data --task disease --cohort cmv

# Inspect a selection without downloading
benchrep-download-data --task disease --cohort rawat-t1d --dry-run
```

Supported cohort names are `malid`, `mitchell-t1d`, `rawat-t1d`, `tb`, `ra`,
and `cmv`. Pooled datasets such as `malid+mitchell-t1d` are a runner-level
combination rather than a cohort of their own, so download their constituent
cohorts. The downloader supports a pinned dataset commit with `--revision`,
resumes through the Hugging Face cache, validates metadata columns and
repertoire presence after download. An optional token can still be supplied
with `--token-env` when using a custom repository that requires one. Existing
files can be checked without network access using
`benchrep-download-data --validate-only`.

Downloaded cohorts retain the Hugging Face repository layout. For example,
Savola RA is written to `data/immunoSEQ/Savola_RA/metadata.tsv` and
`data/immunoSEQ/Savola_RA/repertoires/`; Mal-ID is written to
`data/Mal-ID/metadata.tsv` and `data/Mal-ID/repertoires/`. Tools that support
cohort-native filename templates can consume these paths directly. The
all-method example below creates a uniform compatibility view for evaluators
that expect Mal-ID-style filenames.

The task-specific auxiliary paths are
`data/Mal-ID/vdjdb_minervina_driver_seq_matches.csv` for driver identification
and `data/Mal-ID/scaling_exp_depth_indices_max75k.json.gz` for sequencing-depth
scaling.

## Running the benchmark

### Run every disease-classification benchmark

The tracked example runner downloads the complete public dataset, creates a
symlink-only normalized view for evaluator compatibility, and sequentially runs
all nine methods across all eleven cohort/disease tasks:

```bash
bash examples/run_all_disease_classification.sh
```

The downloaded repertoire files are never copied or modified. For each
metadata row, the runner creates a relative symlink in its run directory using
the filename expected by all evaluators:

```text
source: data/immunoSEQ/Rawat_T1D/repertoires/Brusko_6534_TCRB.tsv.gz
link:   results/all_disease_classification/<run>/staged_data/rawat-t1d/repertoires/part_table_Brusko_6534_Brusko_6534.tsv.gz
```

The staged `metadata.tsv` is a small derived copy. It preserves the source
columns while exposing the common fold column and normalizing `Healthy` (RA)
and `Controller` (TB) to `Healthy/Background`. Relative links contain no
machine-specific absolute paths. A fresh run recreates them on each machine;
they also remain valid if the project tree, including both `data/` and
`results/`, is moved together. Removing a staged run does not remove the
downloaded data, but removing or relocating `data/` by itself breaks its links.

By default it uses the method-specific Conda environments documented below.
The full 99-run matrix is compute-intensive and executes sequentially.
Set `DRY_RUN=1 DOWNLOAD_DATA=0` to print all 99 commands without downloading or
training. `METHODS`, `DATASETS`, `OUTPUT_ROOT`, `USE_GPU`, `N_JOBS`, and
`N_THREADS` can be overridden as environment variables.

For example, run the complete benchmark configuration for all nine methods on
Savola RA:

```bash
DATASETS=ra bash examples/run_all_disease_classification.sh
```

The download step is narrowed to the selected cohorts, so a single-cohort run
fetches only that cohort rather than the full 14.6 GiB dataset.

#### Pooled cohorts

`malid+mitchell-t1d` is the united T1D evaluation: rather than a cohort of its
own, it pools Mitchell specimens into the same three-fold CV split as
Zaslavsky/Mal-ID, so both are trained and tested together with each specimen
keeping its preassigned fold.

```bash
DATASETS=malid+mitchell-t1d bash examples/run_all_disease_classification.sh
```

Both cohorts are read in place from their own staged directories — a pooled
dataset creates no additional symlinks or copies. The runner passes
`--ext_metadata_path`/`--ext_data_dir`/`--ext_file_template` to every method,
and merging is handled by `utils/cohort_merge.py`.

Whenever merging is active, evaluators canonicalize V/J gene labels
(`utils/gene_harmonization.py`); within-cohort runs keep each cohort's native
labels. This is load-bearing rather than cosmetic: Zaslavsky/Mal-ID calls
include the allele (`TRBV13*01`, `TRBV20/OR9-2*02`) while the released
immunoSEQ cohorts are already collapsed to bare gene names (`TRBV13`), so
pooling without it would leave the two cohorts with almost entirely disjoint
gene vocabularies. Canonicalization strips the allele and collapses the
Adaptive-style `-1` suffix on IMGT singleton families; the latter is a no-op on
the released cohorts, which are preprocessed that way already, and matters only
when pooling a differently-prepared cohort of your own.

Mal-ID-Lite additionally falls back to amino-acid clone IDs for pooled runs,
because Zaslavsky/Mal-ID repertoires carry a usable nucleotide CDR3 column and
immunoSEQ-sourced ones do not; the column is dropped so both halves are treated
identically. For that reason a pooled Mal-ID-Lite cache is not
interchangeable with a single-cohort one, and the runner gives it its own
`cache_dir`.

Contrast pooling with the cross-cohort transfer protocol
(`evals/zeroshot_disease_classification.py`), which trains on one cohort and
evaluates on another rather than training on both.

Any evaluator can be pooled with an arbitrary cohort by passing the same
`--ext_*` arguments directly.

### Run every driver-sequence identification benchmark

```bash
bash examples/run_all_driver_identification.sh
```

Six methods across the three diseases the ground-truth match table covers
(HIV, Influenza, COVID-19) — 18 runs — plus the random-chance recall@k baseline
written to `random_baseline_recall.csv` in the run directory. Emerson and
Ensemble Regression take `--k` as space-separated integers while the rest take a
comma-separated list; the runner emits both from a single `K` setting.

The task covers Mal-ID only, because the published driver labels are keyed to
Mal-ID repertoire identifiers. Ensemble XGBoost only scores CDR3s with the models
saved by disease classification and never fits here, so it needs the earlier
run's `OUTPUT_ROOT`:

```bash
OUTPUT_ROOT=results/disease bash examples/run_all_disease_classification.sh
DISEASE_RUN_ROOT=results/disease bash examples/run_all_driver_identification.sh
```

ABMIL, DeepRC and GIANA reuse that run's checkpoints and cluster files when
present and otherwise fit their own, so every method except Ensemble XGBoost
also runs standalone. Without `DISEASE_RUN_ROOT`, Ensemble XGBoost is reported
as skipped rather than producing all-NaN folds. `K`, `DISEASES`, `METHODS`,
`MAX_REPERTOIRES` and `RANDOM_BASELINE` can be overridden.

### Run every demographic-confounding analysis

```bash
bash examples/run_all_demographic_analysis.sh
```

Both analyses of section 4, selected with `ANALYSES`:

- `matched_controls` — seven methods x four diseases x two modes (56 runs).
  `adjust` passes `--adjust_distribution_by_demographics`; `baseline` passes
  `--random_baseline_seeds 7 14 21 28 35`. The step also writes the 24 cohort
  sample lists. GIANA is excluded: its evaluator accepts the matched-cohort flag
  but has no `--random_baseline_seeds`, so it has no paired control.
- `feature_concat` — seven cells x four diseases (28 runs), covering the
  demographics-only baseline, each method on the complete-demographics subset
  via `--require_demographics`, and the concatenated model. CMV comes from the
  Emerson cohort, so this analysis stages two cohorts.

DeepRC and DeepTCR follow the published demographic runs, whose training
schedules differ from those evaluators' argparse defaults; see the `DEEPRC_*`
and `DEEPTCR_*` variables in the script.

### Run every sequencing-depth scaling experiment

```bash
bash examples/run_all_depth_scaling.sh
```

By default, four methods across Lupus and HIV (8 runs), each sweeping every
depth and replicate in the published indices file — 30 complete
cross-validation runs per method and disease. This runner only ever operates
on the Mal-ID dataset, since the indices are keyed to Mal-ID repertoire
identifiers.

Not every method runs by default. `ostmeyer_2019`, `giana_2021`, the
`ensemble_regression`/`ensemble_xgboost` `_kmer`/`_vj` sub-model ablations, and
`malid_lite` are all supported but excluded from the default `METHODS` list —
opt into any of them by setting `METHODS` explicitly, e.g.:

```bash
METHODS="ostmeyer_2019 giana_2021" bash examples/run_all_depth_scaling.sh
METHODS=malid_lite bash examples/run_all_depth_scaling.sh
```

`malid_lite` (this repository's Mal-ID-Lite reimplementation, a separate,
runnable stand-in for the original Mal-ID classifier) is excluded by default
because each (depth, repeat) is a full Mal-ID-Lite training run rather than a
lightweight in-process evaluator. It also behaves differently from every other
method here: rather than one run per (disease, depth, repeat), it runs once
per (depth, repeat), covering every disease in `DISEASES` together via
Mal-ID-Lite's own multi-binary classification mode, and writes one combined
output JSON per (depth, repeat) with a `"disease"` field per result entry,
instead of one JSON per disease. See `evals/mal_id_lite_depth_experiment.py`'s
own module docstring for why (this also matters for correctness, not just
speed: Mal-ID-Lite recomputes clone_id on the actual downsampled sequences for
every (depth, repeat), and every disease is trained against the same
healthy/reference-class participants — sharing one cache and one set of ESM-2
embeddings across all requested diseases means that shared reference
population is only computed once per (depth, repeat), rather than once per
disease. 

The subsampling indices themselves are a precomputed file
(`scaling_exp_depth_indices_max75k.json.gz`) downloaded from Hugging Face
alongside the rest of the dataset, not generated by this runner — depths,
replicate count, minimum sequence count, and seed all come from that file
rather than from flags, so the subsampling is exactly the published one.

It was built once, offline, by `preprocessing/generate_depth_indices.py`, and
is not one shared list of row positions: it holds one independently-drawn
index list *per qualifying repertoire*, each drawn from that repertoire's own
row range (a repertoire with 200k sequences and one with 80k each get their
own 75k-index draw, from `range(0, 200_000)` and `range(0, 80_000)`
respectively). What's shared across repertoires is only the recipe: same
depths, same 5 repetitions, same 75k-sequence eligibility floor, same fixed
master seed (so every user gets bit-identical draws). Nesting falls out of
truncation, not sorting: each repertoire's 75k-index list is drawn once per
repetition and never re-drawn per depth, so depth D just takes that list's
first D entries — depth 1000's rows are therefore always a subset of depth
5000's, which are a subset of depth 10000's, and so on, for that specific
repertoire and repetition.

Because the full matrix is expensive, `DEPTHS` and `REPLICATES` narrow it
without modifying the downloaded indices — they write a filtered copy under the
run directory and tag the output filename, so a narrowed run never collides with
a full one:

```bash
DEPTHS=1000 REPLICATES=1 bash examples/run_all_depth_scaling.sh
DEPTHS=75000 METHODS=ensemble_xgboost bash examples/run_all_depth_scaling.sh
```

The published scaling runs kept XGBoost and GIANA on CPU even where a GPU was
available; `XGBOOST_DEVICE` and `GIANA_USE_GPU` expose that, and `USE_GPU=0`
forces CPU throughout.

Every runner accepts `DRY_RUN=1 DOWNLOAD_DATA=0`, which prints each fully
expanded command without downloading data or training anything.

## Preprocessing

Every cohort is processed through a cohort-specific adapter and converges on
the same AIRR-compliant schema (`cdr3_aa`, `v_call`, `j_call`).

**Zaslavsky/Mal-ID adapter** (`preprocessing/`): drops non-productive
rearrangements and low-confidence V calls, strips IgBLAST whitespace and
uppercases amino-acid fields, drops rows with missing or non-standard CDR3/V/J,
collapses V-gene alleles indistinguishable under the FR3 primer set (per
Meysman et al.), renames to AIRR columns, and splits per-participant tables into
per-specimen files.

**Mitchell, Rawat, Musvosvi, Savola, and Emerson adapters**
(`external_data_process/`): handle each cohort's input schema, rename Adaptive
columns (`aminoAcid` → `cdr3_aa`, etc.), convert Adaptive V/J names (for example,
`TCRBV07-02`) to AIRR/IMGT form (`TRBV7-2`), strip allele annotations, and trim
the flanking conserved cysteine and phenylalanine from each CDR3 to use a
consistent sequence definition. Specimens with fewer than 1,000 unique
post-preprocessing sequences are excluded where required by the cohort
protocol.

**Cross-cohort gene-label reconciliation**: when Zaslavsky/Mal-ID and Mitchell
T1D repertoires are combined, the Adaptive-style `-1` suffix on singleton TRBV
families (TRBV2, TRBV9, TRBV13–15, TRBV18, TRBV19, TRBV27, TRBV28, TRBV30) is
collapsed so the same gene receives the same label in both cohorts.
Within-cohort evaluations preserve each cohort's native labels. Reconciliation
is handled by `utils/gene_harmonization.py`.

Other preprocessing utilities:

- **Depth indices** (`preprocessing/generate_depth_indices.py`) — pre-generates reproducible nested subsampling indices for the depth-scaling experiment.
- **Driver sequence matching** (`preprocessing/process_driver_sequences.py`) — matches VDJdb entries to benchmark repertoires via Levenshtein similarity.
- **Demographic analysis** (`preprocessing/check_demographics.py`) — summarizes demographic completeness per disease.

## Preprint

Currently anonymized.

```
-->
