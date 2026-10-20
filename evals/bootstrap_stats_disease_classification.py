"""
Bootstrap CIs and paired significance tests for the disease-classification table.

Every method in a cohort is evaluated on the same pre-assigned resamples
(``evals.bootstrap_indices_disease_classification``), so each method-vs-method
difference is paired. Writes two CSVs:

    <output_prefix>_metrics.csv
        AUROC and AUPRC per (cohort, method): point estimate, percentile 95% CI
        and bootstrap SD.

    <output_prefix>_pairwise.csv
        Per (cohort, method_a, method_b): delta = AUROC(a) - AUROC(b) with its
        paired-bootstrap CI and p, the DeLong p as a closed-form cross-check, and
        Benjamini-Hochberg q-values for both.

By default every pair of methods within a cohort is tested. --set_a/--set_b
restrict the family to set_a x set_b. BH is applied across exactly the rows
written, so the family on the command line must match what is reported. The 28
reported tests (feature ensembles vs deep learning) are:

    python -m evals.bootstrap_stats_disease_classification \\
        --probabilities_dir <dir> --index_dir <dir> \\
        --cohorts Covid19 HIV Influenza Lupus T1D RA_malid90 Tb \\
        --set_a Ensemble_XGBoost Mal-ID --set_b DeepRC DeepTCR

AUPRC note: resampling is class-stratified, so prevalence is constant across
replicates and the AUPRC random baseline is fixed at n_pos/n (baseline_auprc).
AUPRC is not comparable across cohorts, whose prevalences differ.

--probabilities_dir holds one pooled out-of-fold prediction CSV per method and
disease. Four filename conventions coexist there (see prob_path). Mal-ID files
have their own schema; every other file has specimen_label (or sample_name) and
model_score.
"""

import argparse
import csv
import itertools
import os

import numpy as np

from utils.bootstrap import (
    auprc_from_counts,
    auroc_from_counts,
    load_cohort,
    load_indices,
    normalize_id,
    point_auprc,
    point_auroc,
)
from utils.paired_tests import bh_fdr, delong_test, paired_bootstrap_test


COHORTS = ["Covid19", "HIV", "Influenza", "Lupus", "T1D", "RA", "RA_malid90", "Tb"]
# subset cohort -> parent disease, for locating probability files
COHORT_PARENT = {"RA_malid90": "RA"}
METHODS = [
    "ABMIL",
    "DeepRC",
    "DeepTCR",
    "Emerson",
    "Ensemble",
    "Ensemble_XGBoost",
    "GIANA",
    "Mal-ID",
    "Ostmeyer",
]
# Four filename/disease-token conventions coexist in the probabilities directory.
DOUBLE_US = {"ABMIL", "DeepRC", "DeepTCR", "Ensemble_XGBoost", "GIANA"}
DISEASE_TOKEN_LONG = {"RA": "Rheumatoid_Arthritis", "Tb": "TB_progression"}  # DOUBLE_US
DISEASE_TOKEN_MALID = {"Tb": "TB"}  # Mal-ID; others match the cohort name


def prob_path(prob_dir, method, disease):
    if method in DOUBLE_US:
        return os.path.join(prob_dir, f"{method}__{DISEASE_TOKEN_LONG.get(disease, disease)}.csv")
    if method == "Mal-ID":
        return os.path.join(prob_dir, f"Mal-ID_{DISEASE_TOKEN_MALID.get(disease, disease)}.csv")
    return os.path.join(prob_dir, f"{method}_{disease}.csv")


def positive_class(cohort_rows):
    """The disease_label_str carried by the positive class of this cohort."""
    names = {r["disease_label_str"] for r in cohort_rows if int(r["disease_label"]) == 1}
    if len(names) != 1:
        raise ValueError(f"ambiguous positive class: {names}")
    return names.pop()


def load_scores(prob_dir, method, disease, id_to_row, n_rows, cohort_rows=None):
    """Align one method's positive-class score onto the cohort row order.

    Returns None if the method has no file for this cohort; raises if the join
    leaves any cohort row unfilled, so a silent partial join is impossible.
    """
    path = prob_path(prob_dir, method, disease)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    if method == "Mal-ID":
        # Distinct schema: no disease_label/model_score. The positive-class
        # probability column is named after the class and the two ensemble_P()
        # columns appear in inconsistent order across diseases, so select by name.
        if cohort_rows is None:
            raise ValueError("Mal-ID requires cohort_rows to resolve the positive class")
        pos_name = positive_class(cohort_rows)
        score_col = f"ensemble_P({pos_name})"
        if score_col not in rows[0]:
            raise ValueError(f"{path}: no column {score_col!r} (have {list(rows[0])})")
        id_col, truth_col = "specimen_label", "true_disease"
    else:
        id_col = "specimen_label" if "specimen_label" in rows[0] else "sample_name"
        score_col, truth_col = "model_score", None

    s = np.full(n_rows, np.nan)
    truth = {}
    for x in rows:
        r = id_to_row.get(normalize_id(x[id_col]))
        if r is not None:
            s[r] = float(x[score_col])
            if truth_col:
                truth[r] = x[truth_col]
    if np.isnan(s).any():
        raise ValueError(f"{method}/{disease}: {int(np.isnan(s).sum())} cohort rows unmatched")
    if truth_col:  # cross-check the method file's own labels against the cohort
        bad = [
            r for r, t in truth.items()
            if (t == positive_class(cohort_rows)) != bool(int(cohort_rows[r]["disease_label"]))
        ]
        if bad:
            raise ValueError(f"{method}/{disease}: {len(bad)} label disagreements with cohort")
    return s


def percentile_ci(dist, alpha=0.05):
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def write_csv(path, rows):
    # 6 significant figures, not decimals: DeLong p-values reach 1e-9.
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--probabilities_dir", required=True,
                    help="Directory of per-method, per-disease prediction CSVs.")
    ap.add_argument("--index_dir", required=True,
                    help="Output directory of bootstrap_indices_disease_classification.")
    ap.add_argument("--scheme", choices=["participant", "specimen"], default="participant",
                    help="Resampling unit. Report participant; specimen is a sensitivity check.")
    ap.add_argument("--cohorts", nargs="+", default=COHORTS, choices=COHORTS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--set_a", nargs="+", choices=METHODS,
                    help="Test only set_a x set_b pairs (requires --set_b).")
    ap.add_argument("--set_b", nargs="+", choices=METHODS)
    ap.add_argument("--output_prefix", default="results/bootstrap_disease_classification")
    args = ap.parse_args()
    if bool(args.set_a) != bool(args.set_b):
        ap.error("--set_a and --set_b must be given together")

    metrics, pairwise = [], []
    for cohort in args.cohorts:
        cohort_rows, y, id_to_row = load_cohort(args.index_dir, cohort)
        counts = load_indices(args.index_dir, cohort, args.scheme)
        disease = COHORT_PARENT.get(cohort, cohort)
        n, n_pos = int(y.size), int((y == 1).sum())

        scores, roc, point = {}, {}, {}
        for method in args.methods:
            if cohort == "RA" and method == "Mal-ID":
                continue  # Mal-ID covers 90 of 91 RA repertoires; it appears under RA_malid90
            s = load_scores(args.probabilities_dir, method, disease, id_to_row, n, cohort_rows)
            if s is None:
                continue
            dist_roc = auroc_from_counts(y, s, counts)
            dist_prc = auprc_from_counts(y, s, counts)
            roc_lo, roc_hi = percentile_ci(dist_roc)
            prc_lo, prc_hi = percentile_ci(dist_prc)
            scores[method], roc[method], point[method] = s, dist_roc, point_auroc(y, s)
            metrics.append({
                "cohort": cohort, "method": method, "scheme": args.scheme,
                "n": n, "n_pos": n_pos, "baseline_auprc": n_pos / n,
                "auroc": point[method], "auroc_lo": roc_lo, "auroc_hi": roc_hi,
                "auroc_sd": float(dist_roc.std(ddof=1)),
                "auprc": point_auprc(y, s), "auprc_lo": prc_lo, "auprc_hi": prc_hi,
                "auprc_sd": float(dist_prc.std(ddof=1)),
            })

        if args.set_a:
            pairs = [(a, b) for a in args.set_a for b in args.set_b]
        else:
            pairs = itertools.combinations(sorted(scores), 2)
        for a, b in pairs:
            if a not in scores or b not in scores:
                continue
            r = paired_bootstrap_test(roc[a], roc[b], point[a], point[b])
            d = delong_test(y, scores[a], scores[b])
            pairwise.append({
                "cohort": cohort, "method_a": a, "method_b": b, "scheme": args.scheme,
                "n": n, "auroc_a": point[a], "auroc_b": point[b], **r,
                "delong_se": d["se"], "delong_p": d["p"],
            })

    for p_col, q_col in (("p", "q_bh"), ("delong_p", "delong_q_bh")):
        for r, q in zip(pairwise, bh_fdr([r[p_col] for r in pairwise])):
            r[q_col] = float(q)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    write_csv(f"{args.output_prefix}_metrics.csv", metrics)
    print(f"{len(metrics)} method x cohort cells -> {args.output_prefix}_metrics.csv")
    if not pairwise:
        return
    write_csv(f"{args.output_prefix}_pairwise.csv", pairwise)

    n_boot = sum(r["q_bh"] < 0.05 for r in pairwise)
    n_delong = sum(r["delong_q_bh"] < 0.05 for r in pairwise)
    n_agree = sum((r["q_bh"] < 0.05) == (r["delong_q_bh"] < 0.05) for r in pairwise)
    n_floor = sum(r["p"] <= r["p_floor"] + 1e-12 for r in pairwise)
    print(f"{len(pairwise)} pairwise tests -> {args.output_prefix}_pairwise.csv")
    print(f"  BH q < 0.05: {n_boot} paired bootstrap, {n_delong} DeLong; "
          f"agree on {n_agree}/{len(pairwise)}")
    print(f"  {n_floor} bootstrap p-values sit at the floor {pairwise[0]['p_floor']:.1e}")


if __name__ == "__main__":
    main()
