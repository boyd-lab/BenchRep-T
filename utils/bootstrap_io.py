"""Loading and summarizing helpers shared by the bootstrap scripts.

The resampling and metric code itself is utils.bootstrap; this module only
reads prediction files and index sets and turns per-replicate metrics into
point estimates, bootstrap SDs and percentile 95% intervals.
"""

import csv
from pathlib import Path

import numpy as np

from utils import bootstrap as boot

ROOT = Path(__file__).resolve().parents[1]
PROBS = ROOT / "probs_for_bootstrap"
FOLD = "malid_cross_validation_fold_id_when_in_test_set"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def counts_from_npz(path, n_rows, dtype=np.float32):
    """Replicate multiplicities from a saved participant index set (any file name)."""
    with np.load(path, allow_pickle=False) as z:
        assert int(z["n_rows"]) == n_rows and int(z["n_boot"]) == boot.N_BOOT, path
        flat, indptr = z["flat_idx"], z["indptr"]
    counts = np.zeros((boot.N_BOOT, n_rows), dtype=dtype)
    np.add.at(counts, (np.repeat(np.arange(boot.N_BOOT), np.diff(indptr)), flat.astype(np.int64)), 1)
    return counts


def generated_counts(cohort, name):
    """Class-stratified participant resamples drawn from the per-cohort seed `name`.

    Deterministic, so the same cohort and name always give the same resamples."""
    flat, indptr = boot.participant_indices(cohort, boot.N_BOOT, boot.cohort_rng(name))
    counts = np.zeros((boot.N_BOOT, len(cohort)), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(boot.N_BOOT), np.diff(indptr)), flat.astype(np.int64)), 1)
    return counts


def summarize(y, scores, counts):
    """AUROC and AUPRC point estimate, bootstrap SD and percentile 95% CI.

    Returns (fields, {"auroc": distribution, "auprc": distribution})."""
    fields, dists = {"n": int(y.size), "n_pos": int(y.sum())}, {}
    for metric in ("auroc", "auprc"):
        dist = getattr(boot, f"{metric}_from_counts")(y, scores, counts)
        assert np.isfinite(dist).all()
        lo, hi = np.percentile(dist, [2.5, 97.5])
        fields.update({metric: getattr(boot, f"point_{metric}")(y, scores),
                       f"{metric}_sd": float(dist.std(ddof=1)),
                       f"{metric}_lo": float(lo), f"{metric}_hi": float(hi)})
        dists[metric] = dist
    fields["baseline_auprc"] = float(y.mean())
    return fields, dists


# ── specimen-level score files (participant_label, specimen_label, disease_label,
#    model_score, fold) ───────────────────────────────────────────────────────
def load_predictions(path):
    """Score file keyed by normalized specimen id."""
    table = {}
    for row in read_csv(path):
        sid = boot.normalize_id(row["specimen_label"])
        if sid in table:
            raise ValueError(f"{path}: duplicate specimen {sid}")
        if not np.isfinite(float(row["model_score"])) or int(row["disease_label"]) not in (0, 1):
            raise ValueError(f"{path}: invalid score or label for {sid}")
        table[sid] = row
    return table


def canonical(table):
    """Cohort rows (utils.bootstrap.COHORT_FIELDS) in the order of a score table."""
    return [dict(row_index=i, specimen_id=sid, specimen_id_as_written=row["specimen_label"],
                 participant_label=row["participant_label"],
                 disease_label=int(row["disease_label"]),
                 disease_label_str=row["disease_label_str"], fold=int(row[FOLD]))
            for i, (sid, row) in enumerate(table.items())]


def align(table, cohort, path):
    """Scores in cohort row order; labels, participants and folds must agree."""
    scores = []
    for row in cohort:
        other = table[row["specimen_id"]]
        if (int(other["disease_label"]) != int(row["disease_label"])
                or other["participant_label"] != row["participant_label"]
                or ("fold" in row and int(other[FOLD]) != int(row["fold"]))):
            raise ValueError(f"{path}: labels, participant or fold disagree for {row['specimen_id']}")
        scores.append(float(other["model_score"]))
    return np.array(scores)
