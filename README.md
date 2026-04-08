# AIRRBench

Benchmarking framework for TCR repertoire-based disease classification methods, evaluated on the Mal-ID cohort.

## Overview

AIRRBench provides a standardized comparison of computational methods that predict disease status from T-cell receptor beta (TRB) repertoire sequencing data. The benchmark covers a range of approaches — from statistical association tests to deep learning — and evaluates them on shared cross-validation splits across multiple diseases (HIV, Covid-19, Influenza, Lupus, T1D, and others).

Beyond disease classification, the framework includes experiments for driver sequence identification, sequencing depth sensitivity, external dataset generalization, and demographic confounding analysis.

## Repository Structure

```
models/                          Classification methods
├── emerson_2017.py              Emerson et al. 2017
├── ostmeyer_2019.py             Ostmeyer et al. 2019
├── ensemble_regression.py       Gapped k-mer + V/J gene ensemble
├── ensemble_abmil.py            Attention-based deep MIL
├── GIANA/                       Zhang et al. 2021 (GIANA 4.1)
├── DeepRC/                      Widrich et al. 2020
└── DeepTCR/                     Sidhom et al. 2021
evals/                           Evaluation harness and experiments
preprocessing/                   Data cleaning and preparation
utils/                           Repertoire I/O and gene name harmonization
external_data_process/           External dataset conversion (Adaptive -> AIRR)
```

## Implemented Methods

### Statistical

- **Emerson et al. 2017** — Identifies disease-associated CDR3 sequences via Fisher's exact test, then scores repertoires using a Beta-Binomial generative model over the discovered sequences.

### Feature Engineering

- **Ostmeyer et al. 2019** — Multiple instance learning over 4-mer motifs extracted from CDR3 sequences, encoded with Atchley factors and classified by logistic regression with random restarts.
- **Ensemble Regression** — Weighted combination of two logistic regression models: one over gapped 4-mer frequencies from CDR3 sequences, the other over V/J gene usage frequencies. Hyperparameters (regularization strength, ensemble weight) are tuned via internal cross-validation.

### Deep Learning

- **GIANA (Zhang et al. 2021)** — Isometric encoding of CDR3 sequences into fixed-length vectors, followed by similarity-based clustering. Test repertoires are scored by the disease fraction of clusters their sequences fall into.
- **DeepRC (Widrich et al. 2020)** — 1D-CNN sequence embedding aggregated via modern Hopfield networks for end-to-end repertoire classification.
- **DeepTCR (Sidhom et al. 2021)** — Deep learning framework operating on CDR3 sequences with V/D/J gene features for supervised repertoire classification.
- **Attention-Based MIL (ABMIL)** — Learned amino acid and V/J gene embeddings fed through a 1D-CNN encoder, aggregated via gated attention over each repertoire's sequences.

## Experiments

### Disease Classification

The primary evaluation task. Each method is trained to distinguish a target disease from healthy/background controls in a binary classification setting. Evaluation uses pre-defined 3-fold cross-validation with a 90/10 train/validation split for hyperparameter tuning. A **demographics-only baseline** (logistic regression on age, sex, ancestry) is included to quantify confounding.

Metrics: AUROC, AUPR.

### Driver Sequence Identification

Tests whether models can recover known disease-associated TCR sequences. Ground truth is derived from VDJdb (confidence score >= 2), matched to repertoires via Levenshtein similarity (>= 90%). Currently supported for Emerson 2017 (ranked by Fisher's test p-value) and Ensemble Regression (ranked by decision function score).

Metrics: recall@k (macro-averaged across repertoires).

### Sequencing Depth Sensitivity

Evaluates how classification performance scales with repertoire size. Models are tested at subsampled depths of 1K, 5K, 10K, 25K, 50K, and 75K sequences, with 5 independent repetitions per depth using pre-computed nested subsampling indices for reproducibility.

### External Cohort Generalization

Trains on all internal Mal-ID data and evaluates on independently collected external datasets. External repertoires (originally in Adaptive/immunoSEQ format) are converted to AIRR format with gene name harmonization.

### Demographic Confounding

A stacking experiment to test whether TCR-derived predictions and demographic features carry orthogonal signal. A logistic regression meta-model is trained on the combination of a base model's out-of-sample predictions and demographic features (age, sex, ancestry). If the meta-model substantially outperforms the base model alone, it suggests demographic confounding in the classification signal.

## Data

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

A tab-separated `metadata.tsv` file provides sample annotations:

| Column | Description |
|--------|-------------|
| `participant_label` | Participant identifier |
| `specimen_label` | Specimen identifier |
| `disease` | Disease label (e.g., HIV, Covid19, Healthy/Background) |
| `malid_cross_validation_fold_id_when_in_test_set` | Pre-assigned CV fold (0, 1, or 2) |
| `age`, `sex`, `ancestry` | Demographics |

### Directory Layout

```
data/
├── metadata.tsv                     Sample annotations and CV fold assignments
├── malid_clean/TCR/                 Per-specimen repertoire files (.tsv.gz)
├── public_clones/                   VDJdb ground truth driver sequences per disease
├── depth_indices_seed7.json.gz      Pre-computed subsampling indices
├── external_raw/                    External repertoires (Adaptive format)
└── external_processed/              External repertoires (converted to AIRR)
```

## Preprocessing

The `preprocessing/` directory contains scripts for preparing raw data:

- **Sequence QC** (`clean_tcr_data.py`) — Filters non-productive and low-confidence sequences, removes sequences with non-standard amino acids, standardizes V gene nomenclature, and validates against a reference gene set.
- **Format conversion** (`clean_tcr_data_to_airr.py`) — Maps internal column names to AIRR standard.
- **Specimen splitting** (`clean_airr_split_by_specimen.py`) — Splits per-participant files into per-specimen files, retaining only specimens present in metadata.
- **Depth indices** (`generate_depth_indices.py`) — Pre-generates reproducible subsampling indices for the sequencing depth experiment.
- **Driver sequence matching** (`process_driver_sequences.py`) — Matches VDJdb entries to Mal-ID repertoires via Levenshtein similarity for the driver identification evaluation.
- **Demographic analysis** (`check_demographics.py`) — Summarizes demographic completeness per disease.

Gene name harmonization between Adaptive/immunoSEQ and AIRR/IMGT conventions is handled by `utils/gene_harmonization.py`.

## Setup

Requires Python >= 3.10. Install with [uv](https://docs.astral.sh/uv/):

```bash
# Core dependencies (Emerson, Ostmeyer, Ensemble Regression)
uv sync

# With specific model extras
uv sync --extra giana       # GIANA (adds faiss-cpu)
uv sync --extra abmil       # Attention-based MIL (adds PyTorch)
uv sync --extra deeptcr     # DeepTCR (adds TensorFlow, umap-learn, etc.)
uv sync --extra drivers     # Driver sequence identification (adds python-Levenshtein)

# Everything
uv sync --all-extras
```

Alternatively, with pip:

```bash
pip install -e .             # core only
pip install -e ".[all]"      # all extras
```

For GPU-accelerated models (ABMIL, DeepTCR), ensure CUDA-compatible versions of PyTorch or TensorFlow are installed for your system.
