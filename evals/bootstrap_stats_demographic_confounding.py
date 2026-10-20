"""
Bootstrap SEs and paired tests for the demographic-confounding table.

All seven cells of a disease (demographics only; VJ (Reg), DeepTCR and ABMIL
each with and without demographics) are evaluated on the same pre-assigned
resamples (``evals.bootstrap_indices_demographic_confounding``), so every
difference between cells is paired. Point estimates are pooled across CV folds,
matching each evaluator's headline number. Writes three CSVs:

    <output_prefix>_metrics.csv
        AUROC and AUPRC with bootstrap SE for every cell.

    <output_prefix>_demographics_added.csv
        delta = (TCR + demographics) - (TCR only) per method, with paired SE.

    <output_prefix>_vs_demographics_only.csv
        delta = (TCR only) - (demographics only) per method; positive means the
        repertoire beats the confounders alone. Paired-bootstrap CI and p, BH q
        within each metric (the same 12 contrasts measured two ways, so AUROC and
        AUPRC are separate families), and the DeLong p for AUROC.

DeLong is a closed-form cross-check only. It assumes independent observations,
so it is exact for Covid19 (211 specimens / 211 donors) and CMV (542/542) and
mildly anti-conservative for Lupus (229/228) and HIV (253/251).
"""

import argparse
import csv
import os

import numpy as np

from evals.bootstrap_indices_demographic_confounding import DISEASES
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


METHODS = ["VJ (Reg)", "DeepTCR", "ABMIL"]
STEM = {"VJ (Reg)": "VJ", "DeepTCR": "DeepTCR", "ABMIL": "ABMIL"}
SETTINGS = [("TCR only", "repertoire_only"), ("TCR + demo", "repertoire_plus_demographics")]
BASELINE = "demographics_only"


def load_scores(prob_dir, disease, stem, y, id_to_row):
    """Align one cell's score onto cohort row order; raise on an incomplete join."""
    path = os.path.join(prob_dir, disease, f"{stem}.csv")
    scores = np.full(y.size, np.nan, dtype=np.float64)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            row = id_to_row.get(normalize_id(r["specimen_label"]))
            if row is None:
                continue  # outside the common cohort
            scores[row] = float(r["model_score"])
            if int(r["disease_label"]) != int(y[row]):
                raise ValueError(f"{disease}/{stem}: label disagreement at row {row}")
    missing = int(np.isnan(scores).sum())
    if missing:
        raise ValueError(f"{disease}/{stem}: {missing} cohort rows unscored")
    return scores


def write_csv(path, rows):
    # 6 significant figures, not decimals: DeLong p-values can be tiny.
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--probabilities_dir", required=True,
                    help="Directory with one <disease>/ subdirectory of cell CSVs.")
    ap.add_argument("--index_dir", required=True,
                    help="Output directory of bootstrap_indices_demographic_confounding.")
    ap.add_argument("--scheme", choices=["participant", "specimen"], default="participant",
                    help="Resampling unit. Report participant; specimen is a sensitivity check.")
    ap.add_argument("--output_prefix", default="results/bootstrap_demographic_confounding")
    args = ap.parse_args()

    cells, added = [], []
    vs_demo = {"AUROC": [], "AUPRC": []}
    for disease in DISEASES:
        _, y, id_to_row = load_cohort(args.index_dir, disease)
        counts = load_indices(args.index_dir, disease, args.scheme)
        n, n_pos = int(y.size), int(y.sum())

        def evaluate(label, setting, stem):
            s = load_scores(args.probabilities_dir, disease, stem, y, id_to_row)
            res = {
                "scores": s,
                "AUROC": (auroc_from_counts(y, s, counts), point_auroc(y, s)),
                "AUPRC": (auprc_from_counts(y, s, counts), point_auprc(y, s)),
            }
            cells.append({
                "disease": disease, "method": label, "setting": setting,
                "n": n, "n_pos": n_pos, "prevalence": n_pos / n,
                "auroc": res["AUROC"][1], "auroc_se": float(res["AUROC"][0].std(ddof=1)),
                "auprc": res["AUPRC"][1], "auprc_se": float(res["AUPRC"][0].std(ddof=1)),
            })
            return res

        base = evaluate("Demographics only", "demographics only", BASELINE)
        for method in METHODS:
            tcr, both = [evaluate(method, setting, f"{STEM[method]}__{suffix}")
                         for setting, suffix in SETTINGS]
            added.append({
                "disease": disease, "method": method, "n": n,
                "d_auroc": both["AUROC"][1] - tcr["AUROC"][1],
                "d_auroc_se": float((both["AUROC"][0] - tcr["AUROC"][0]).std(ddof=1)),
                "d_auprc": both["AUPRC"][1] - tcr["AUPRC"][1],
                "d_auprc_se": float((both["AUPRC"][0] - tcr["AUPRC"][0]).std(ddof=1)),
            })
            for metric in ("AUROC", "AUPRC"):
                (dist_a, point_a), (dist_b, point_b) = tcr[metric], base[metric]
                rec = {
                    "disease": disease, "metric": metric, "method": method,
                    "scheme": args.scheme, "n": n,
                    "tcr_only": point_a, "demographics_only": point_b,
                    **paired_bootstrap_test(dist_a, dist_b, point_a, point_b),
                }
                if metric == "AUROC":
                    dl = delong_test(y, tcr["scores"], base["scores"])
                    rec["delong_se"], rec["delong_p"] = dl["se"], dl["p"]
                vs_demo[metric].append(rec)

    for rows in vs_demo.values():
        for r, q in zip(rows, bh_fdr([r["p"] for r in rows])):
            r["q_bh"] = float(q)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    write_csv(f"{args.output_prefix}_metrics.csv", cells)
    write_csv(f"{args.output_prefix}_demographics_added.csv", added)
    # AUROC rows come first and carry the DeLong columns; AUPRC rows leave them blank.
    write_csv(f"{args.output_prefix}_vs_demographics_only.csv",
              vs_demo["AUROC"] + vs_demo["AUPRC"])

    for metric, rows in vs_demo.items():
        n_fav = sum(r["delta"] > 0 for r in rows)
        n_raw = sum(r["p"] < 0.05 for r in rows)
        n_adj = sum(r["q_bh"] < 0.05 for r in rows)
        print(f"{metric} TCR only vs demographics only: {n_fav}/{len(rows)} favour the "
              f"repertoire; {n_raw}/{len(rows)} significant at 0.05 raw, {n_adj} after BH")
    print(f"Wrote {args.output_prefix}_{{metrics,demographics_added,vs_demographics_only}}.csv")


if __name__ == "__main__":
    main()
