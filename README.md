# AIRRBench

Benchmarking framework for adaptive immune receptor repertoire (AIRR) based disease classification.

## Overview

This repository evaluates and compares computational methods that use T-cell receptor (TCR) repertoire sequencing data to predict disease status. Methods are benchmarked on a common evaluation framework using standardized cross-validation splits and metrics.

## Structure

- `models/` — Implementations of repertoire-based classification methods
- `evals/` — Evaluation harness and cross-validation logic
- `preprocessing/` — Scripts for preparing raw AIRR-format data
- `utils/` — Shared utilities

## Implemented Models

- **Emerson 2017** — Fisher's exact test with Beta-Binomial scoring
- **Ostmeyer 2019** — Multiple instance learning with 4-mer motifs
- **GIANA 2020** — Isometric CDR3 encoding with FAISS clustering

## Evaluation

Models are evaluated using pre-defined 3-fold cross-validation with AUROC and AUPR as primary metrics.
