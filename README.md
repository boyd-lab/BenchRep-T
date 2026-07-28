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
└── DeepTCR/                     Sidhom et al. 2021           (deep learning)
evals/                           Per-method experiment scripts (disease, drivers, depth, demographics)
preprocessing/                   Repertoire cleaning and preparation
external_data_process/           Cohort-specific conversion to AIRR and gene harmonization
utils/                           Repertoire I/O, metric helpers, cohort/covariate adjustment
scripts/                         Misc analysis helpers
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

## Tasks

### 1. Disease Classification

Binary disease-vs-control classification uses fixed three-fold
cross-validation, with an internal 80/20 or 90/10 train/validation split for
hyperparameter tuning. The following cohort protocols reuse the same evaluation
harness:

- **Zaslavsky/Mal-ID** — HIV, Lupus, Influenza, COVID-19, and T1D are scored
  against Healthy/Background controls using preassigned folds.
- **Mitchell and Rawat T1D** — Each cohort supports within-cohort evaluation.
  T1D models can also be trained on one cohort and evaluated on the others to
  measure cross-cohort generalization. Zaslavsky/Mal-ID and Mitchell T1D
  repertoires can additionally be pooled for three-fold evaluation.
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

Re-runs disease classification after downsampling every repertoire (training and test) to a common target depth D ∈ {1k, 5k, 10k, 25k, 50k, 75k}. Subsampling indices are pre-generated with a fixed seed and **nested** across depths, so the D sequences at depth D are always a subset of those used at any larger depth — differences across depths cannot be explained by drawing different sequences. Five independent replicates per depth; specimens with fewer than 75k unique sequences are excluded.

### 4. Demographic Confounding Analysis

Two complementary analyses probe whether classifier signal is carried by participant demographics rather than disease biology:

1. **Demographic-matched controls** — For each disease, the random pool of healthy controls is replaced with a subset matched to the disease cohort's dominant confounder (age for Lupus, Influenza, COVID-19; African ancestry for HIV). Matched-control AUROC is compared against a random-control baseline averaged over five draws of the same size. Sample lists are dumped by `dump_demographic_cohorts.py`.
2. **Demographic-feature concatenation** — Each method's repertoire-level features are concatenated with age, sex, and ancestry; the concatenated model is compared against the base model and a demographics-only logistic-regression baseline. Implemented for Ensemble Regression/XGBoost, ABMIL, and DeepTCR (`*_demographics_disease_classification.py`); standalone demographics-only and V/J-gene-only baselines are also provided.

## Data

### Sources

| Cohort | Prediction task | Labeled repertoires | Class composition |
|--------|-----------------|--------------------:|-------------------|
| Zaslavsky/Mal-ID | HIV, T1D, Lupus, COVID-19, and Influenza vs Healthy/Background | 550 | 197 Healthy/Background; 98 HIV; 96 T1D; 64 Lupus; 58 COVID-19; 37 Influenza |
| Mitchell | T1D vs control | 197 | T1D and control |
| Rawat | T1D vs Healthy/Background | 614 | 426 T1D; 188 Healthy/Background |
| Musvosvi | TB progression vs control | 140 | TB progressor and control |
| Savola | Rheumatoid Arthritis vs control | 94 | RA and control |
| Emerson | CMV seropositive vs seronegative | 761 | 340 CMV-positive; 421 CMV-negative |

The Emerson source metadata also contains 25 specimens with unknown CMV status;
these are retained in the metadata but excluded from supervised classification.
Demographic annotations are used where they are available.

### Format

Repertoire files are gzip-compressed TSV (`.tsv.gz`) in AIRR standard format. The key columns used by all methods are:

| Column | Description |
|--------|-------------|
| `cdr3_aa` | CDR3 amino acid sequence |
| `v_call` | V gene (IMGT nomenclature, e.g., TRBV7-2\*01) |
| `j_call` | J gene |
| `duplicate_count` | Clone count (optional; assumed 1 if absent) |

File naming convention: `part_table_<participant>_<specimen>.tsv.gz`

### Metadata

| Cohort | Metadata file |
|--------|---------------|
| Zaslavsky/Mal-ID | `data/malid_clean/metadata.tsv` |
| Mitchell | `data/external_datasets/T1D/metadata_T1D_final.tsv` |
| Rawat | `data/external_datasets/Rawat_T1D/cohort1_metadata.tsv` |
| Musvosvi | `data/external_datasets/tuberculosis/metadata_Tb_final.tsv` |
| Savola | `data/external_datasets/rheumatoid_arthritis/metadata_RA_final.tsv` |
| Emerson | `data/external_datasets/CMV/emerson_cohort_metadata.tsv` |

The harmonized metadata tables use the following common columns:

| Column | Description |
|--------|-------------|
| `participant_label` | Participant identifier |
| `specimen_label` | Specimen identifier |
| `disease` | Disease label (e.g., HIV, Covid19, Healthy/Background) |
| `malid_cross_validation_fold_id_when_in_test_set` | Pre-assigned CV fold (0, 1, or 2) |
| `age`, `sex`, `ancestry` | Demographics, where available |

### HuggingFace

Preprocessed BenchRep-T repertoires and metadata, including Rawat T1D and
Emerson CMV, are hosted at:

**[https://huggingface.co/datasets/lordim/BenchRep-T](https://huggingface.co/datasets/lordim/BenchRep-T)**


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

## Setup

Requires Python >= 3.10. The recommended installation uses one minimal Conda
environment per method family so that DeepTCR's TensorFlow 2.12 constraints do
not conflict with the newer PyTorch and scientific-Python stacks.

| Environment | File | Methods |
|-------------|------|---------|
| `simple` | `environment-simple.yml` | Emerson, Ostmeyer, Ensemble Regression/XGBoost, and ABMIL |
| `giana` | `environment-giana.yml` | GIANA |
| `deeprc` | `environment-deeprc.yml` | DeepRC |
| `deeptcr` | `environment-deeptcr.yml` | DeepTCR |

The environments are separated as follows:

- **`simple`** uses Python 3.11 and contains the shared scientific stack,
  PyTorch, and XGBoost. It runs Emerson, Ostmeyer, Ensemble Regression,
  Ensemble XGBoost, and ABMIL, as well as the common preprocessing and analysis
  utilities. This replaces the former general-purpose `airr-bench`
  environment.
- **`giana`** uses Python 3.11 with Biopython and FAISS CPU. Keeping GIANA
  separate avoids coupling its FAISS and NumPy compatibility constraints to
  the deep-learning environments.
- **`deeprc`** uses Python 3.10 with PyTorch, HDF5, TensorBoard, the upstream
  `deeprc` package, and `widis-lstm-tools`. The PyTorch wheel supplies the CUDA
  runtime used by the GPU launchers.
- **`deeptcr`** uses Python 3.10 and the legacy-compatible TensorFlow 2.12,
  NumPy 1.23, pandas 1.5, scikit-learn 1.2, and Biopython 1.76 stack required by
  the bundled DeepTCR TensorFlow-1 compatibility and deprecated Biopython APIs.
  It is intentionally isolated from the newer scientific-Python environments.

Create the environments from the repository root:

```bash
conda env create -f environment-simple.yml
conda env create -f environment-giana.yml
conda env create -f environment-deeprc.yml
conda env create -f environment-deeptcr.yml
```

Install BenchRep-T itself into each environment in editable mode. Dependencies
are already supplied by the corresponding YAML, so `--no-deps` prevents pip
from changing the tested method-specific pins:

```bash
for env_name in simple giana deeprc deeptcr; do
    conda run -n "${env_name}" python -m pip install --no-deps -e .
done
```

Run the method-level import smoke tests with:

```bash
bash scripts/smoke_test_conda_envs.sh
```

The environment previously called `airr-bench` is now named `simple`. Existing
local `airr-bench` environments do not need to be removed; new installations
should use `environment-simple.yml`.

For GPU execution, PyTorch includes its CUDA runtime dependencies. DeepTCR is
pinned to TensorFlow 2.12 because the bundled DeepTCR source uses TensorFlow 1
compatibility APIs. Its GPU launchers configure the node's CUDA/cuDNN paths and
the additional Hopper compatibility settings needed on H100/H200 GPUs.

### uv or pip

The same dependency groups are exposed as optional extras in `pyproject.toml`.
Install with [uv](https://docs.astral.sh/uv/):

```bash
# Core dependencies
uv sync

# With specific model extras
uv sync --extra simple
uv sync --extra giana
uv sync --extra deeprc
uv sync --extra deeptcr
uv sync --extra drivers

# Everything
uv sync --all-extras
```

Alternatively, with pip:

```bash
pip install -e .             # core only
pip install -e ".[all]"      # all extras
```

<!-- removing for anonymity
## Preprint

The BenchRep-T preprint is available [here](https://www.biorxiv.org/content/10.64898/2026.06.09.727013v1.abstract) with accompanying citation:

```bibtex
@article{im2026benchrep,
  title={BenchRep-T: A Systematic Evaluation of T-Cell Repertoire-Based Disease Diagnostics},
  author={Im, Chiho and Cohen-Lavi, Liel and Buendia, Alejandro and Kundaje, Anshul and Boyd, Scott D},
  journal={bioRxiv},
  pages={2026--06},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
-->
