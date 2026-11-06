"""Bootstrap CIs and paired tests for disease classification, all tasks.

Every method of a cohort is scored on the same B=10,000 class-stratified
participant resamples (utils.bootstrap), so method differences are paired.

Cohorts
  three_fold  Covid19, HIV, Influenza, Lupus, T1D, RA (the 90 specimens Mal-ID
              scores), TB: pooled out-of-fold predictions, saved index sets.
  held_out    CMV (Emerson validation cohort) and Rawat T1D (fold 2).
  zero_shot   Rawat-trained models on Mal-ID and Mitchell T1D.
              held_out and zero_shot average repeat specimens per participant.
  cmv_rawat_3fold
              Appendix 3-fold CV on CMV and Rawat T1D. CMV excludes P00207
              (884 sequences after preprocessing, dropped by Mal-ID), so all nine
              methods share 760 participants.

Paired tests: VJ-kmer (XG) and Mal-ID vs. DeepRC and DeepTCR on the nine
benchmark tasks (36 tests), paired bootstrap and DeLong, each BH-adjusted
across the 36.

Writes metrics.csv, distributions.npz and paired_tests.csv to --output-dir.

    python -m evals.bootstrap_disease_classification [--output-dir DIR]
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evals.bootstrap_stats_disease_classification import COHORT_PARENT, load_scores
from utils import bootstrap as boot
from utils.bootstrap_io import (FOLD, PROBS, ROOT, align, canonical, counts_from_npz,
                                generated_counts, load_predictions, read_csv, summarize,
                                write_csv)
from utils.paired_tests import bh_fdr, delong_test, paired_bootstrap_test

DC = PROBS / "disease_classification"
METHODS = ["Emerson", "Ostmeyer", "GIANA", "DeepRC", "DeepTCR", "ABMIL",
           "VJ-kmer (Reg)", "VJ-kmer (XG)", "Mal-ID"]

# ── three_fold: saved probability files and index sets ───────────────────────
THREE_FOLD = ["Covid19", "HIV", "Influenza", "Lupus", "T1D", "RA_malid90", "Tb"]
THREE_FOLD_FILES = {"Emerson": "Emerson", "Ostmeyer": "Ostmeyer", "GIANA": "GIANA",
                    "DeepRC": "DeepRC", "DeepTCR": "DeepTCR", "ABMIL": "ABMIL",
                    "VJ-kmer (Reg)": "Ensemble", "VJ-kmer (XG)": "Ensemble_XGBoost",
                    "Mal-ID": "Mal-ID"}

# ── held_out / zero_shot: (cohort, folder, positive class) ───────────────────
HELD_OUT = {"CMV": ("cmv_fold2", "saved_probabilities", "CMV"),
            "Rawat_T1D": ("rawat_trained_t1d", "rawat_fold2", "T1D")}
ZERO_SHOT = {"MalID_clean_T1D": ("rawat_trained_t1d", "malid_clean_all_folds", "T1D"),
             "Mitchell_T1D": ("rawat_trained_t1d", "mitchell_all_folds", "T1D")}
HELD_OUT_FILES = {"Emerson": "Emerson", "Ostmeyer": "Ostmeyer", "GIANA": "GIANA",
                  "DeepRC": "DeepRC", "DeepTCR": "DeepTCR", "ABMIL": "ABMIL",
                  "VJ-kmer (Reg)": "Ensemble_regression", "VJ-kmer (XG)": "Ensemble_XGBoost",
                  "Mal-ID": "Mal-ID"}

# ── cmv_rawat_3fold: score files of the 3-fold runs ──────────────────────────
CMV_RUN = ROOT / "results/CMV_3fold_cv/20261029_031819"
RAWAT_RUNS = [ROOT / "results/Rawat_T1D_3fold_cv/20260725_225435",
              ROOT / "results/Rawat_T1D_3fold_cv/20261029_005231_missing_methods"]
EXCLUDED = {"CMV": {"P00207"}, "Rawat": set()}
RUN_FILES = {"Emerson": "emerson", "Ostmeyer": "ostmeyer", "GIANA": "giana",
             "DeepRC": "deeprc", "DeepTCR": "deeptcr", "ABMIL": "abmil",
             "VJ-kmer (Reg)": "ensemble_regression", "VJ-kmer (XG)": "ensemble_xgboost"}
MALID_3FOLD = {
    "CMV": (CMV_RUN / "checkpoints/malid_lite/CMV_vs_Healthy_Background/ensemble_predictions.csv",
            "CMV"),
    "Rawat": (DC / "rawat_trained_t1d/rawat_fold2/Mal-ID.csv", "T1D"),
}

PAIRS = [(a, b) for a in ("VJ-kmer (XG)", "Mal-ID") for b in ("DeepRC", "DeepTCR")]
PAIRED_COHORTS = THREE_FOLD + list(HELD_OUT)


def participant_scores(path, cohort, positive, fold2):
    """Scores averaged over each participant's specimens, in cohort row order."""
    rows = read_csv(path)
    if fold2:
        rows = [r for r in rows if r.get("fold_id", r.get(FOLD, "2")) == "2"]
    by_participant = collections.defaultdict(list)
    for row in rows:
        if "model_score" in row:
            score, truth = float(row["model_score"]), int(row.get("disease_label", row.get("label", "-1")))
        else:
            assert row.get("abstained", "False").lower() == "false", path
            score, truth = float(row[f"ensemble_P({positive})"]), int(row["true_disease"] == positive)
        assert truth in (0, 1) and np.isfinite(score), path
        by_participant[row["participant_label"]].append((score, truth))
    assert set(by_participant) == {r["participant_label"] for r in cohort}, (path, "coverage")
    scores = []
    for row in cohort:
        entries = by_participant[row["participant_label"]]
        assert {t for _, t in entries} == {int(row["disease_label"])}, (path, row)
        assert len(entries) == int(row["n_repertoires"]), (path, "specimen count")
        scores.append(np.mean([s for s, _ in entries]))
    return np.array(scores)


def three_fold():
    """(task, cohort, y, {method: scores}, counts) for the saved 3-fold cohorts."""
    idx = str(DC / "bootstrap_indices")
    for name in THREE_FOLD:
        rows, y, ids = boot.load_cohort(idx, name)
        counts = boot.load_indices(idx, name, "participant")
        scores = {m: load_scores(str(DC / "probabilities"), f, COHORT_PARENT.get(name, name),
                                 ids, len(y), rows) for m, f in THREE_FOLD_FILES.items()}
        yield "three_fold", name, y.astype(int), scores, counts


def participant_level(task, cohorts):
    for name, (collection, folder, positive) in cohorts.items():
        idx = DC / collection / "bootstrap_indices"
        cohort = read_csv(idx / f"{name}__cohort.csv")
        y = np.array([int(r["disease_label"]) for r in cohort])
        counts = counts_from_npz(idx / f"{name}__participant.npz", len(cohort))
        lower = folder.endswith("all_folds")
        scores = {m: participant_scores(DC / collection / folder / f"{f.lower() if lower else f}.csv",
                                        cohort, positive, fold2=name == "Rawat_T1D")
                  for m, f in HELD_OUT_FILES.items()}
        yield task, name, y, scores, counts


def malid_3fold(dataset, emerson):
    """Mal-ID ensemble predictions as a score table; folds must match the other methods."""
    source, positive = MALID_3FOLD[dataset]
    data = pd.read_csv(source, dtype={"specimen_label": str, "participant_label": str})
    assert not data["abstained"].astype(str).eq("True").any(), source
    table = {}
    for _, r in data.iterrows():
        sid = boot.normalize_id(r["specimen_label"])
        assert str(r["fold_id"]) == emerson[sid][FOLD], (source, sid, "fold")
        table[sid] = {"participant_label": r["participant_label"],
                      "specimen_label": r["specimen_label"],
                      "disease_label": int(r["true_disease"] == positive),
                      "disease_label_str": r["true_disease"], FOLD: str(r["fold_id"]),
                      "model_score": float(r[f"ensemble_P({positive})"])}
    return table


def cmv_rawat_3fold():
    for dataset in ("CMV", "Rawat"):
        runs = [CMV_RUN] if dataset == "CMV" else RAWAT_RUNS
        tables = {}
        for method, stem in RUN_FILES.items():
            paths = [r / "scores" / f"{stem}.csv" for r in runs if (r / "scores" / f"{stem}.csv").exists()]
            assert len(paths) == 1, (dataset, method, paths)
            tables[method] = load_predictions(paths[0])
        tables["Mal-ID"] = malid_3fold(dataset, tables["Emerson"])
        for method, table in tables.items():
            tables[method] = {k: v for k, v in table.items()
                              if v["specimen_label"] not in EXCLUDED[dataset]}
        assert len({frozenset(t) for t in tables.values()}) == 1, f"{dataset}: coverage differs"
        cohort = canonical(tables["Emerson"])
        counts = generated_counts(cohort, f"democonfound:{dataset}_3fold")
        y = np.array([r["disease_label"] for r in cohort])
        scores = {m: align(tables[m], cohort, f"{dataset}/{m}") for m in METHODS}
        yield "cmv_rawat_3fold", f"{dataset}_3fold", y, scores, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "results/bootstrap/disease_classification")
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics, dists, paired = [], {}, []
    sources = [three_fold(), participant_level("held_out", HELD_OUT),
               participant_level("zero_shot", ZERO_SHOT), cmv_rawat_3fold()]
    for source in sources:
        for task, name, y, scores, counts in source:
            auroc = {}
            for method in METHODS:
                s = scores[method]
                fields, d = summarize(y, s, counts)
                np.testing.assert_allclose(fields["auroc"], roc_auc_score(y, s), atol=1e-7)
                metrics.append(dict(task=task, cohort=name, method=method, **fields))
                dists.update({f"{name}__{method}__{k}": v for k, v in d.items()})
                auroc[method] = (fields["auroc"], d["auroc"])
            if name in PAIRED_COHORTS:
                for a, b in PAIRS:
                    test = paired_bootstrap_test(auroc[a][1], auroc[b][1], auroc[a][0], auroc[b][0])
                    dl = delong_test(y, scores[a], scores[b])
                    paired.append(dict(cohort=name, method_a=a, method_b=b, n=len(y),
                                       auroc_a=auroc[a][0], auroc_b=auroc[b][0], **test,
                                       delong_p=dl["p"], delong_se=dl["se"]))
            print(f"{task:16s} {name:16s} n={len(y)}", flush=True)
    paired = pd.DataFrame(paired)
    paired["q_boot"] = bh_fdr(paired.p)
    paired["q_delong"] = bh_fdr(paired.delong_p)
    write_csv(out / "metrics.csv", metrics)
    np.savez_compressed(out / "distributions.npz", **dists)
    paired.to_csv(out / "paired_tests.csv", index=False)
    print(f"{len(metrics)} cells, {len(paired)} paired tests "
          f"({(paired.q_boot < .05).sum()} significant) -> {out}")


if __name__ == "__main__":
    main()
