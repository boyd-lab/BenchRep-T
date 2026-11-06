"""Bootstrap CIs and paired tests for demographics vs. repertoire models (Fig. 5c).

Seven settings per dataset (CELLS): demographics only, and VJ (Reg), DeepTCR
and ABMIL with TCR only or TCR + demographics. Every setting of a dataset is
scored on the same specimens and the same B=10,000 class-stratified
participant resamples (utils.bootstrap).

Datasets
  Lupus, HIV, Covid19   pooled 3-fold predictions, saved index sets
                        (probs_for_bootstrap/demographics_confound).
  Rawat, CMV            held-out fold 2 (Rawat fold 2; Emerson validation cohort).
  Influenza, TB, RA     pooled 3-fold predictions.
For the last five, resamples are drawn from a fixed per-dataset seed.

Paired tests: each TCR-only model vs. demographics only on the eight datasets
(24 tests), paired bootstrap, BH-adjusted across the 24.

Writes metrics.csv, distributions.npz and paired_tests.csv to --output-dir.

    python -m evals.bootstrap_demographics [--output-dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import bootstrap as boot
from utils.bootstrap_io import (FOLD, PROBS, ROOT, align, canonical, generated_counts,
                                load_predictions, summarize, write_csv)
from utils.paired_tests import bh_fdr, paired_bootstrap_test

CELLS = ["demographics_only", "vj_only", "vj_demo", "deeptcr_only", "deeptcr_demo",
         "abmil_only", "abmil_demo"]
TCR_ONLY = {"vj_only": "VJ (Reg)", "deeptcr_only": "DeepTCR", "abmil_only": "ABMIL"}

SAVED = PROBS / "demographics_confound"
SAVED_FILES = dict(zip(CELLS, [
    "demographics_only.csv", "VJ__repertoire_only.csv", "VJ__repertoire_plus_demographics.csv",
    "DeepTCR__repertoire_only.csv", "DeepTCR__repertoire_plus_demographics.csv",
    "ABMIL__repertoire_only.csv", "ABMIL__repertoire_plus_demographics.csv"]))
HELD_OUT_RUN = ROOT / "results/rawat_cmv_demographics/20261105_combined_cmv_corrected"
THREE_FOLD_RUN = ROOT / "results/influenza_tb_ra_demographics/20261102"
# dataset -> (run directory or None for saved files, evaluation fold or None for all)
DATASETS = {"HIV": (None, None), "Lupus": (None, None), "Covid19": (None, None),
            "Influenza": (THREE_FOLD_RUN, None), "TB": (THREE_FOLD_RUN, None),
            "RA": (THREE_FOLD_RUN, None), "CMV": (HELD_OUT_RUN, "2"), "Rawat": (HELD_OUT_RUN, "2")}


def run_cohort(run, dataset, fold):
    """Cohort rows from a run's prepared metadata, restricted to the evaluation fold."""
    prepared = run / dataset / "prepared"
    target = json.loads((prepared / "cohort.json").read_text())["target_disease"]
    data = pd.read_csv(prepared / "metadata.tsv", sep="\t", dtype=str)
    if fold is not None:
        data = data[data[FOLD].eq(fold)]
    table = {boot.normalize_id(r.specimen_label): dict(
                 participant_label=r.participant_label, specimen_label=r.specimen_label,
                 disease_label=int(r.disease == target), disease_label_str=r.disease,
                 **{FOLD: r[FOLD]})
             for _, r in data.iterrows()}
    return canonical(table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/bootstrap/demographics")
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics, dists, paired = [], {}, []
    for dataset, (run, fold) in DATASETS.items():
        if run is None:
            cohort, _, _ = boot.load_cohort(str(SAVED / "bootstrap_indices"), dataset)
            counts = boot.load_indices(str(SAVED / "bootstrap_indices"), dataset, "participant")
            paths = {cell: SAVED / dataset / f for cell, f in SAVED_FILES.items()}
        else:
            cohort = run_cohort(run, dataset, fold)
            seed = f"democonfound:{dataset}_fold2" if fold else f"democonfound:{dataset}_3fold"
            counts = generated_counts(cohort, seed)
            paths = {cell: run / dataset / cell / "scores.csv" for cell in CELLS}
        y = np.array([int(r["disease_label"]) for r in cohort])
        auroc = {}
        for cell, path in paths.items():
            table = load_predictions(path)
            # Saved demographics-only files cover more specimens; scored on the common set.
            if not (run is None and cell == "demographics_only"):
                assert set(table) == {r["specimen_id"] for r in cohort}, (path, "coverage")
            fields, d = summarize(y, align(table, cohort, path), counts)
            metrics.append(dict(dataset=dataset, cell=cell, **fields))
            dists.update({f"{dataset}__{cell}__{k}": v for k, v in d.items()})
            auroc[cell] = (fields["auroc"], d["auroc"])
        for cell, label in TCR_ONLY.items():
            test = paired_bootstrap_test(auroc[cell][1], auroc["demographics_only"][1],
                                         auroc[cell][0], auroc["demographics_only"][0])
            paired.append(dict(dataset=dataset, method=label, n=len(y),
                               auroc_tcr=auroc[cell][0], auroc_demo=auroc["demographics_only"][0],
                               **test))
        print(f"{dataset:10s} {len(cohort)} specimens", flush=True)
    paired = pd.DataFrame(paired)
    paired["q"] = bh_fdr(paired.p)
    write_csv(out / "metrics.csv", metrics)
    np.savez_compressed(out / "distributions.npz", **dists)
    paired.to_csv(out / "paired_tests.csv", index=False)
    print(f"{len(metrics)} cells, {len(paired)} paired tests ({(paired.q < .05).sum()} significant) -> {out}")


if __name__ == "__main__":
    main()
