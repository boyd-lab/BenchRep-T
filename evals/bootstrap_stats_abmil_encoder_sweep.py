"""
Bootstrap SEs and paired tests for the ABMIL pretrained-encoder sweep on Covid19.

The sweep is evaluated on the same Covid19 cohort as the disease-classification
table (255 specimens, 58 Covid19 / 197 healthy, pooled three-fold out-of-fold
predictions), so it reuses that table's pre-assigned Covid19 index set rather
than drawing new resamples. Every row is therefore directly comparable to the
main table's Covid19 column, and each encoder is paired against the
no-pretraining baseline: plain ABMIL, taken from the main collection.

--sweep_dir holds <stem>.csv for every encoder in ENCODERS and, optionally, the
sweep's own source_metrics_summary.csv. When that file is present the recomputed
point estimates are checked against it. p-values are uncorrected (3 tests).
"""

import argparse
import csv
import os

import numpy as np

from evals.bootstrap_stats_disease_classification import load_scores, percentile_ci, write_csv
from utils.bootstrap import (
    auprc_from_counts,
    auroc_from_counts,
    load_cohort,
    load_indices,
    normalize_id,
    point_auprc,
    point_auroc,
)
from utils.paired_tests import paired_bootstrap_test


COHORT = "Covid19"
# display name -> filename stem in the sweep directory (None = main collection ABMIL)
ENCODERS = {
    "ABMIL (no pretraining)": None,
    "ABMIL + ESM-2": "ESM2",
    "ABMIL + TCR-BERT": "TCRBERT",
    "ABMIL + TCRvalid": "TCRvalid",
}
BASELINE = "ABMIL (no pretraining)"
# display name -> encoder key in source_metrics_summary.csv
SUMMARY_KEY = {
    "ABMIL (no pretraining)": "baseline",
    "ABMIL + ESM-2": "esm2",
    "ABMIL + TCR-BERT": "tcrbert",
    "ABMIL + TCRvalid": "tcrvalid",
}


def load_sweep_scores(sweep_dir, stem, id_to_row, n_rows, cohort_rows, y):
    """Align one encoder's scores onto the cohort row order.

    Specimen labels here keep the zero-padded form (`M64-010`) where the cohort
    table stores the normalised one (`M64-10`), affecting 88 of 255 rows, so the
    same normalisation used elsewhere is applied. Raises on a partial join.
    """
    with open(os.path.join(sweep_dir, f"{stem}.csv")) as fh:
        rows = list(csv.DictReader(fh))
    s = np.full(n_rows, np.nan)
    for r in rows:
        i = id_to_row.get(normalize_id(r["specimen_label"]))
        if i is None:
            continue
        s[i] = float(r["model_score"])
        if int(r["disease_label"]) != int(y[i]):
            raise ValueError(f"{stem}: label disagreement at {r['specimen_label']}")
        if int(r["malid_cross_validation_fold_id_when_in_test_set"]) != int(cohort_rows[i]["fold"]):
            raise ValueError(f"{stem}: fold disagreement at {r['specimen_label']}")
    if np.isnan(s).any():
        raise ValueError(f"{stem}: {int(np.isnan(s).sum())} of {n_rows} cohort rows unmatched")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sweep_dir", required=True,
                    help="Directory of per-encoder Covid19 prediction CSVs.")
    ap.add_argument("--probabilities_dir", required=True,
                    help="Main disease-classification prediction CSVs (for the ABMIL baseline).")
    ap.add_argument("--index_dir", required=True,
                    help="Output directory of bootstrap_indices_disease_classification.")
    ap.add_argument("--scheme", choices=["participant", "specimen"], default="participant",
                    help="Resampling unit. Report participant; specimen is a sensitivity check.")
    ap.add_argument("--output_csv", default="results/bootstrap_abmil_encoder_sweep.csv")
    args = ap.parse_args()

    cohort_rows, y, id_to_row = load_cohort(args.index_dir, COHORT)
    counts = load_indices(args.index_dir, COHORT, args.scheme)
    n, n_pos = int(y.size), int((y == 1).sum())

    rows, roc = [], {}
    for disp, stem in ENCODERS.items():
        if stem is None:
            s = load_scores(args.probabilities_dir, "ABMIL", COHORT, id_to_row, n, cohort_rows)
        else:
            s = load_sweep_scores(args.sweep_dir, stem, id_to_row, n, cohort_rows, y)
        roc[disp] = auroc_from_counts(y, s, counts)
        prc = auprc_from_counts(y, s, counts)
        roc_lo, roc_hi = percentile_ci(roc[disp])
        prc_lo, prc_hi = percentile_ci(prc)
        rows.append({
            "model": disp, "scheme": args.scheme, "n": n, "n_pos": n_pos,
            "auroc": point_auroc(y, s), "auroc_lo": roc_lo, "auroc_hi": roc_hi,
            "auroc_sd": float(roc[disp].std(ddof=1)),
            "auprc": point_auprc(y, s), "auprc_lo": prc_lo, "auprc_hi": prc_hi,
            "auprc_sd": float(prc.std(ddof=1)),
        })

    # paired difference against the no-pretraining baseline, on the shared resamples
    base = next(r for r in rows if r["model"] == BASELINE)
    for r in rows:
        if r is base:
            r.update(d_auroc="", d_auroc_lo="", d_auroc_hi="", d_auroc_sd="", p="")
            continue
        c = paired_bootstrap_test(roc[r["model"]], roc[BASELINE], r["auroc"], base["auroc"])
        r.update(d_auroc=c["delta"], d_auroc_lo=c["lo"], d_auroc_hi=c["hi"],
                 d_auroc_sd=c["boot_sd"], p=c["p"])

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_csv(args.output_csv, rows)
    for r in rows:
        d = "" if r is base else f"  dAUROC={r['d_auroc']:+.3f} ± {r['d_auroc_sd']:.3f} (p={r['p']:.3f})"
        print(f"{r['model']:24} AUROC={r['auroc']:.3f} ± {r['auroc_sd']:.3f}  "
              f"AUPRC={r['auprc']:.3f} ± {r['auprc_sd']:.3f}{d}")
    print(f"Wrote {args.output_csv}")

    summary = os.path.join(args.sweep_dir, "source_metrics_summary.csv")
    if os.path.exists(summary):
        with open(summary) as fh:
            saved = {r["encoder"]: r for r in csv.DictReader(fh)}
        print("validation vs source_metrics_summary.csv:")
        for r in rows:
            s = saved[SUMMARY_KEY[r["model"]]]
            da = abs(r["auroc"] - float(s["overall_AUROC"]))
            dp = abs(r["auprc"] - float(s["overall_AUPR"]))
            print(f"  {r['model']:24} |dAUROC|={da:.1e}  |dAUPRC|={dp:.1e}")


if __name__ == "__main__":
    main()
