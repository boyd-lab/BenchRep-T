# AIRRBench

Benchmarking framework for adaptive immune receptor repertoire (AIRR) based disease classification.

## Overview

This repository evaluates and compares computational methods that use T-cell receptor (TCR) repertoire sequencing data to predict disease status. Methods are benchmarked on a common evaluation framework using pre-defined 3-fold cross-validation splits, with AUROC and AUPR as primary metrics.

## Repository Structure

```
models/            Repertoire-based classification methods
  DeepRC/          Modern Hopfield network (Widrich et al. 2020)
  DeepTCR/         Deep learning TCR framework (Sidhom et al. 2021)
  GIANA/           Isometric CDR3 encoding (Liu et al. 2021)
evals/             Evaluation harness, cross-validation, and experiments
preprocessing/     Raw data cleaning and preparation scripts
utils/             Shared I/O and gene harmonization utilities
external_data_process/  External dataset preprocessing
```

## Implemented Models

| Model | File | Approach |
|-------|------|----------|
| **Emerson 2017** | `models/emerson_2017.py` | Fisher's exact test for diagnostic TCR discovery + Beta-Binomial generative scoring |
| **Ostmeyer 2019** | `models/ostmeyer_2019.py` | Multiple instance learning with 4-mer motifs, Atchley factor encoding, logistic regression |
| **GIANA** | `models/GIANA/` | Isometric CDR3 encoding + similarity-based clustering + per-cluster disease fraction scoring |
| **Ensemble Regression** | `models/ensemble_regression.py` | Gapped 4-mer + V/J gene frequency features, weighted ensemble of two logistic regression models |
| **Ensemble ABMIL** | `models/ensemble_abmil.py` | Attention-based deep MIL with learned AA embeddings, 1D-CNN encoder, gated attention aggregation |
| **DeepRC** | `models/DeepRC/` | 1D-CNN sequence embedding + modern Hopfield network aggregation (end-to-end differentiable) |
| **DeepTCR** | `models/DeepTCR/` | Deep learning framework supporting supervised classification with CDR3 + V/D/J gene features |

All models follow a common API: `load_repertoire()`, `preload_repertoires()`, `clear_cache()`, `train()`, `predict_diagnosis()`.

## Evaluation Suite

### Disease Classification

Each evaluator loads metadata, constructs binary (disease vs. Healthy/Background) labels, and runs 3-fold CV with a 90/10 train/validation split for hyperparameter tuning.

| Evaluator | Model | Notes |
|-----------|-------|-------|
| `Emerson2017Evaluator` | Emerson 2017 | Tunes p-value threshold on validation set |
| `Ostmeyer2019Evaluator` | Ostmeyer 2019 | Configurable restarts, abundance method (A or B) |
| `GIANAEvaluator` | GIANA | Clusters train+test sequences, scores test by cluster disease fraction |
| `EnsembleRegressionEvaluator` | Ensemble Regression | Supports `ensemble`, `kmer_only`, `vj_only` sub-models |
| `ABMILEvaluator` | Ensemble ABMIL | GPU-accelerated, early stopping |
| `DeepRC2020Evaluator` | DeepRC | GPU-accelerated, reads AIRR .tsv.gz directly |
| `DeepTCREvaluator` | DeepTCR | GPU-accelerated, loads sequences in memory |
| `DemographicFeaturesEvaluator` | Logistic regression | Baseline using age, sex, ancestry only (confounding control) |

### Driver Sequence Identification

Evaluates whether models can recover known disease-associated TCR sequences (ground truth from VDJdb, score >= 2, Levenshtein similarity >= 90%).

| Evaluator | Model | Scoring |
|-----------|-------|---------|
| `Emerson2017DriverIdentificationEvaluator` | Emerson 2017 | Ranks CDR3s by Fisher's exact test p-value |
| `EnsembleRegressionDriverIdentificationEvaluator` | Ensemble Regression | Ranks CDR3s by decision function score |

Metrics: precision@k and recall@k, macro-averaged across repertoires.

### Additional Experiments

| Experiment | File | Description |
|------------|------|-------------|
| **Sequencing Depth** | `evals/sequencing_depth_experiment.py` | Evaluates all models at depths 1K–75K sequences (5 repetitions per depth) using pre-computed subsampling indices |
| **External Evaluation** | `evals/external_evaluation.py` | Trains on all internal data, evaluates on external datasets (Adaptive format converted to AIRR) |
| **Meta-Model** | `evals/meta_model_repertoire_demographics_disease_classification.py` | Stacks TCR model predictions with demographic features to test orthogonality of signal sources |

## Data Format

### Repertoire Files

Gzip-compressed TSV (`.tsv.gz`) with AIRR-standard columns:

- `cdr3_aa` — CDR3 amino acid sequence
- `v_call` — V gene call (IMGT nomenclature, e.g., TRBV7-2*01)
- `j_call` — J gene call
- `duplicate_count` — Sequence count (optional; defaults to 1)

File naming: `part_table_<participant>_<specimen>.tsv.gz`

### Metadata

Tab-separated file (`metadata.tsv`) with columns:

- `participant_label`, `specimen_label` — Sample identifiers
- `disease` — Disease label (e.g., HIV, Lupus, T1D, Covid19, Influenza, Healthy/Background)
- `malid_cross_validation_fold_id_when_in_test_set` — Pre-defined CV fold (0, 1, or 2)
- `age`, `sex`, `ancestry` — Demographics (used by baseline and meta-model evaluators)

### Expected Directory Layout

```
data/
├── metadata.tsv
├── malid_clean/TCR/              Per-specimen repertoire files
├── public_clones/                VDJdb ground truth driver sequences
├── depth_indices_seed7.json.gz   Pre-computed subsampling indices
├── external_raw/                 External repertoires (Adaptive format)
└── external_processed/           Converted external repertoires (AIRR format)
```

## Preprocessing

| Script | Purpose |
|--------|---------|
| `clean_tcr_data.py` | Filters non-productive/low-confidence sequences, standardizes V gene names, validates against reference |
| `clean_tcr_data_to_airr.py` | Converts internal column names to AIRR standard format |
| `clean_airr_split_by_specimen.py` | Splits per-participant files into per-specimen files, filtering to metadata-matched specimens |
| `generate_depth_indices.py` | Pre-generates reproducible subsampling indices (seed=7) for sequencing depth experiments |
| `process_driver_sequences.py` | Matches VDJdb ground truth to Mal-ID repertoires via Levenshtein similarity (>= 90%) |
| `check_demographics.py` | Analyzes demographic completeness per disease, generates summary plots |

## Utilities

- **`utils/repertoire_io.py`** — Unified repertoire loading with optional subsampling (by indices, count, or fraction)
- **`utils/gene_harmonization.py`** — Adaptive-to-AIRR gene name conversion, allele stripping for cross-dataset compatibility

## Setup

**Python**: 3.10+ (conda environment `antibody_py310`)

**Core dependencies**: pandas, numpy, scipy, scikit-learn, tqdm

**Deep learning** (optional, for ABMIL/DeepRC/DeepTCR): PyTorch with CUDA support

**Additional**: Levenshtein (driver sequence matching), matplotlib/seaborn (plots), faiss-cpu (GIANA clustering)
