# BenchRep-T: A Systematic Evaluation of T-Cell Repertoire-Based Disease Diagnostics

A unified benchmark for TCRβ repertoire-based disease classification,
harmonizing the Mal-ID cohort (Zaslavsky et al. 2025) with external T1D
(Mitchell and Rawat), TB, RA, and Emerson CMV cohorts into a single
AIRR-compliant schema. BenchRep-T evaluates statistical, feature-engineered,
clustering, and deep-learning methods using identical inputs and fixed splits.

## Overview

BenchRep-T includes disease and serostatus tasks organized into four groups:

- **Mal-ID-only**: HIV, Lupus, Influenza, COVID-19
- **Multi-cohort T1D**: Mal-ID, Mitchell, and Rawat cohorts, supporting
  within-cohort evaluation and transfer to independently collected cohorts
- **External disease cohorts**: Tuberculosis progression and Rheumatoid
  Arthritis
- **CMV serostatus**: Emerson et al. CMV-positive versus CMV-negative
  repertoires

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
preprocessing/                   Mal-ID data cleaning and preparation
external_data_process/           immunoSEQ dataset conversion (Adaptive -> AIRR) and gene harmonization
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

- **Mal-ID-only diseases** (HIV, Lupus, Influenza, COVID-19) — Mal-ID's pre-assigned 3-fold splits, scored against Healthy/Background controls.
- **Multi-cohort T1D** — Mal-ID T1D specimens can be pooled with the Mitchell
  cohort for three-fold evaluation. The Rawat cohort adds a larger independent
  T1D dataset for within-Rawat cross-validation and out-of-distribution
  evaluation on Mal-ID and Mitchell.
- **External-only diseases (TB, RA)** — Each cohort is evaluated entirely within itself under the same 3-fold protocol; this measures within-cohort accuracy on a different assay rather than cross-cohort transfer.
- **CMV serostatus** — The Emerson cohort is evaluated as CMV-positive versus
  CMV-negative classification using fixed three-fold splits.

Metrics: AUROC, AUPR.

### 2. Driver Sequence Identification

Tests whether per-sequence scores produced as a by-product of classification surface known antigen-specific TCRs. Ground truth is built from VDJdb (confidence ≥ 2), augmented for COVID-19 with experimentally validated SARS-CoV-2 clonotypes from Minervina et al., and matched to Mal-ID repertoires via ≥90% Levenshtein similarity. Supported for Emerson 2017, GIANA, Ensemble Regression, Ensemble XGBoost, ABMIL, and DeepRC; each ranks sequences by its native score (Fisher p-value, cluster disease fraction, decision-function weight, attention weight, etc.). A random-chance baseline is provided by `compute_random_baseline_recall.py`.

Metrics: recall@k (k ∈ {100, 1000, 10000}, macro-averaged across disease-positive repertoires).

### 3. Sequencing Depth Scaling

Re-runs disease classification after downsampling every repertoire (training and test) to a common target depth D ∈ {1k, 5k, 10k, 25k, 50k, 75k}. Subsampling indices are pre-generated with a fixed seed and **nested** across depths, so the D sequences at depth D are always a subset of those used at any larger depth — differences across depths cannot be explained by drawing different sequences. Five independent replicates per depth; specimens with fewer than 75k unique sequences are excluded.

### 4. Demographic Confounding Analysis

Two complementary analyses probe whether classifier signal is carried by participant demographics rather than disease biology:

1. **Demographic-matched controls** — For each disease, the random pool of healthy controls is replaced with a subset matched to the disease cohort's dominant confounder (age for Lupus, Influenza, COVID-19; African ancestry for HIV). Matched-control AUROC is compared against a random-control baseline averaged over five draws of the same size. Sample lists are dumped by `dump_demographic_cohorts.py`.
2. **Demographic-feature concatenation** — Each method's repertoire-level features are concatenated with age, sex, and ancestry; the concatenated model is compared against the base model and a demographics-only logistic-regression baseline. Implemented for Ensemble Regression/XGBoost, ABMIL, and DeepTCR (`*_demographics_disease_classification.py`); standalone demographics-only and V/J-gene-only baselines are also provided.

## Data

### Sources

- **Mal-ID** (Zaslavsky et al. 2025) — 550 TCRβ specimens from 542 participants across multiple clinical sites: 197 Healthy/Background, 98 HIV, 96 T1D, 64 Lupus, 58 COVID-19, 37 Influenza. A subset has age, sex, and self-reported ancestry annotations.
- **Mitchell T1D** — 197 independently collected T1D and control repertoires.
- **Rawat T1D** — 614 repertoires: 426 T1D and 188 Healthy/Background.
- **TB progression** (Musvosvi et al.) — 140 repertoires.
- **Rheumatoid Arthritis** (Savola et al.) — 94 repertoires.
- **Emerson CMV** (Emerson et al. 2017) — 761 labeled repertoires: 340
  CMV-positive and 421 CMV-negative. An additional 25 specimens with unknown
  CMV status are retained in the source metadata but excluded from supervised
  classification.

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

`metadata_malid.tsv` provides Mal-ID sample annotations;
`metadata_{T1D,RA,Tb}_final.tsv` provides the corresponding external-cohort
annotations; `data/external_datasets/Rawat_T1D/cohort1_metadata.tsv` describes
Rawat T1D; and
`data/external_datasets/CMV/emerson_cohort_metadata.tsv` describes Emerson CMV.
Common columns include:

| Column | Description |
|--------|-------------|
| `participant_label` | Participant identifier |
| `specimen_label` | Specimen identifier |
| `disease` | Disease label (e.g., HIV, Covid19, Healthy/Background) |
| `malid_cross_validation_fold_id_when_in_test_set` | Pre-assigned CV fold (0, 1, or 2) |
| `age`, `sex`, `ancestry` | Demographics (Mal-ID; subset of specimens) |

### HuggingFace

Preprocessed BenchRep-T repertoires and metadata, including Rawat T1D and
Emerson CMV, are hosted at:

**[https://huggingface.co/datasets/lordim/BenchRep-T](https://huggingface.co/datasets/lordim/BenchRep-T)**


## Preprocessing

Mal-ID and the external cohorts enter the pipeline in different raw formats and
follow conceptually parallel cleanup paths that converge on a single
AIRR-compliant schema (`cdr3_aa`, `v_call`, `j_call`).

**Mal-ID** (`preprocessing/`): drops non-productive rearrangements and low-confidence V calls, strips IgBLAST whitespace and uppercases amino-acid fields, drops rows with missing or non-standard CDR3/V/J, collapses V-gene alleles indistinguishable under the FR3 primer set (per Meysman et al.), renames to AIRR columns, and splits per-participant tables into per-specimen files.

**External cohorts** (`external_data_process/`): cohort-specific adapters handle
the Mitchell, Rawat, TB, RA, and Emerson input schemas. They rename Adaptive
columns (`aminoAcid` → `cdr3_aa`, etc.), convert Adaptive V/J names (for example,
`TCRBV07-02`) to AIRR/IMGT form (`TRBV7-2`), strip allele annotations, and trim
the flanking conserved cysteine and phenylalanine from each CDR3 to match the
Mal-ID definition. Specimens with fewer than 1,000 unique post-preprocessing
sequences are excluded where required by the cohort protocol.

**Cross-source gene-label reconciliation**: when an immunoSEQ cohort is merged with Mal-ID (T1D), the Adaptive-style `-1` suffix on singleton TRBV families (TRBV2, TRBV9, TRBV13–15, TRBV18, TRBV19, TRBV27, TRBV28, TRBV30) is collapsed so the same gene receives the same label across sources. Within-source evaluations preserve each source's native labels. Reconciliation is handled by `utils/gene_harmonization.py`.

Other preprocessing utilities:

- **Depth indices** (`preprocessing/generate_depth_indices.py`) — pre-generates reproducible nested subsampling indices for the depth-scaling experiment.
- **Driver sequence matching** (`preprocessing/process_driver_sequences.py`) — matches VDJdb entries to Mal-ID repertoires via Levenshtein similarity.
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
