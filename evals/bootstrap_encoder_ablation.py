"""Bootstrap CIs and paired tests for the ABMIL pretrained-encoder ablation.

Baseline ABMIL (per-fold-trained CNN encoder, the main-table predictions) vs.
ABMIL with frozen ESM-2, TCR-BERT and TCRvalid encoders, on the nine tasks.
Each task reuses the disease-classification index set (B=10,000), so every
encoder is paired against the baseline on identical resamples. Delta = encoder
AUROC minus baseline AUROC per resample; two-sided p-values from the tail
counts, unadjusted:
    p_raw       = min(1, 2*min(#delta<=0, #delta>=0)/B)
    p_corrected = min(1, 2*(min(#delta<=0, #delta>=0)+1)/(B+1))
RA is scored on all 91 specimens.

Writes summary.csv and distributions.npz to --output-dir.

    python -m evals.bootstrap_encoder_ablation [--output-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np

from utils.bootstrap import normalize_id
from utils.bootstrap_io import PROBS, ROOT, counts_from_npz, read_csv, summarize, write_csv

DC = PROBS / "disease_classification"
SWEEPS = ROOT / "results/abmil_encoder_sweeps"
ENCODERS = {"baseline": "ABMIL (no pretraining)", "esm2": "ABMIL + ESM-2",
            "tcrbert": "ABMIL + TCR-BERT", "tcrvalid": "ABMIL + TCRvalid"}
# dataset -> (cohort in the disease-classification index set, sweep run prefix,
#             token of the baseline ABMIL probability file)
TASKS = {
    "HIV": ("HIV", "hiv", "HIV"),
    "Lupus": ("Lupus", "lupus", "Lupus"),
    "Covid19": ("Covid19", None, "Covid19"),
    "Influenza": ("Influenza", None, "Influenza"),
    "T1D": ("T1D", "pooled_t1d", "T1D"),
    "TB": ("Tb", "tb_participant_resplit", "TB_progression"),
    "RA": ("RA", "ra", "Rheumatoid_Arthritis"),
    "CMV": (None, "cmv", None),
    "Rawat": (None, None, None),
}


def inputs(dataset):
    cohort, prefix, token = TASKS[dataset]
    if cohort:
        cohort_path = DC / f"bootstrap_indices/cohorts/{cohort}.csv"
        index_path = DC / f"bootstrap_indices/indices/{cohort}__participant.npz"
        paths = {"baseline": DC / f"probabilities/ABMIL__{token}.csv"}
    else:
        base, name = (("cmv_fold2", "CMV") if dataset == "CMV" else ("rawat_trained_t1d", "Rawat_T1D"))
        cohort_path = DC / base / f"bootstrap_indices/{name}__cohort.csv"
        index_path = DC / base / f"bootstrap_indices/{name}__participant.npz"
        paths = {"baseline": DC / base / ("saved_probabilities/ABMIL.csv" if dataset == "CMV"
                                          else "rawat_fold2/ABMIL.csv")}
    for encoder in ("esm2", "tcrbert", "tcrvalid"):
        if dataset == "Covid19":
            stem = {"esm2": "ESM2", "tcrbert": "TCRBERT", "tcrvalid": "TCRvalid"}[encoder]
            paths[encoder] = PROBS / f"encoder_sweep/Covid19/{stem}.csv"
            continue
        if dataset in ("Influenza", "Rawat"):
            run = ("influenza" if dataset == "Influenza" else "rawat_t1d") + "_stage1_abmil_config05_seed7_comparison"
        else:
            run = f"{prefix}_covid_encoder_settings_seed7"
        run_dir = SWEEPS / run / encoder
        assert (run_dir / "COMPLETE").exists(), f"incomplete run: {run_dir}"
        paths[encoder] = run_dir / "scores.csv"
    return cohort_path, index_path, paths


def align_scores(path, cohort):
    """Scores in cohort row order, keyed by specimen (or participant for held-out sets)."""
    key = "specimen_id" if "specimen_id" in cohort[0] else "participant_label"
    source_key = "specimen_label" if key == "specimen_id" else key
    rows = {}
    for row in read_csv(path):
        identity = normalize_id(row[source_key])
        assert identity not in rows, (path, "duplicate", identity)
        rows[identity] = row
    assert set(rows) == {normalize_id(r[key]) for r in cohort}, (path, "coverage")
    scores = []
    for row in cohort:
        source = rows[normalize_id(row[key])]
        assert int(source["disease_label"]) == int(row["disease_label"]), (path, row[key])
        assert source["participant_label"] == row["participant_label"], (path, row[key])
        if "fold" in row:
            assert int(source["malid_cross_validation_fold_id_when_in_test_set"]) == int(row["fold"])
        scores.append(float(source["model_score"]))
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/bootstrap/encoder_ablation")
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows, dists = [], {}
    for dataset in TASKS:
        cohort_path, index_path, paths = inputs(dataset)
        cohort = read_csv(cohort_path)
        assert [int(r["row_index"]) for r in cohort] == list(range(len(cohort)))
        y = np.array([int(r["disease_label"]) for r in cohort])
        # float64 counts, as in the original encoder-sweep bootstrap.
        counts = counts_from_npz(index_path, len(cohort), dtype=np.float64)
        n_participants = len({r["participant_label"] for r in cohort})
        for encoder, path in paths.items():
            fields, d = summarize(y, align_scores(path, cohort), counts)
            row = dict(dataset=dataset, encoder=encoder, model=ENCODERS[encoder],
                       n_participants=n_participants, **fields)
            if encoder == "baseline":
                base_point, base_dist = fields["auroc"], d["auroc"]
            else:
                delta = d["auroc"] - base_dist
                lo, hi = np.percentile(delta, [2.5, 97.5])
                tail = min(int((delta <= 0).sum()), int((delta >= 0).sum()))
                row.update(delta_auroc=fields["auroc"] - base_point,
                           delta_auroc_sd=float(delta.std(ddof=1)),
                           delta_auroc_lo=float(lo), delta_auroc_hi=float(hi),
                           p_raw=min(1., 2 * tail / len(delta)),
                           p_corrected=min(1., 2 * (tail + 1) / (len(delta) + 1)))
                d["delta_auroc"] = delta
            rows.append(row)
            dists.update({f"{dataset}__{encoder}__{k}": v for k, v in d.items()})
        print(f"{dataset:10s} {len(cohort)} specimens, {n_participants} participants", flush=True)
    write_csv(out / "summary.csv", rows)
    np.savez_compressed(out / "distributions.npz", **dists)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
