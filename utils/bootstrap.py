"""
Pre-assigned bootstrap index sets, and AUROC/AUPRC evaluated on every replicate
at once.

One index set is drawn per cohort and shared by every method, so per-method CIs
land on identical resamples and a method-vs-method difference can be
bootstrapped by subtracting the two per-replicate metric vectors (paired
bootstrap; see ``utils.paired_tests``). Comparisons must reuse the saved index
sets rather than drawing fresh resamples, or the pairing is lost.

Index sets point into a canonical cohort table (``cohorts/<name>.csv``, one row
per specimen, ``row_index`` column), not into any individual method's
prediction CSV. Two schemes are written per cohort:

    'participant' (primary)
        Cluster bootstrap: resample participants with replacement and carry all
        of a participant's specimens, since repeat specimens from one donor are
        not independent. Replicate length varies, so it is stored CSR-style.

    'specimen' (sensitivity check)
        Resample specimen rows with replacement. Fixed length. Understates CI
        width wherever a donor contributes more than one specimen.

Both are stratified by class: each replicate draws exactly the observed number
of units per class, so prevalence is fixed and AUROC is always defined.
"""

import collections
import csv
import hashlib
import os
import re

import numpy as np


MASTER_SEED = 20260725
N_BOOT = 10_000

COHORT_FIELDS = [
    "row_index",
    "specimen_id",
    "specimen_id_as_written",
    "participant_label",
    "disease_label",
    "disease_label_str",
    "fold",
]


def normalize_id(s):
    """Strip zero-padding from a trailing numeric suffix.

    Emerson/Ensemble/Ostmeyer write DenverT1D-009 where the other methods write
    DenverT1D-9 (40 of 489 T1D specimens). Cohort tables store the normalized
    form, so specimen ids must pass through this before joining.
    """
    return re.sub(r"-0+(\d+)$", r"-\1", s)


def cohort_rng(name, master_seed=MASTER_SEED):
    """Deterministic per-cohort RNG. Stable across runs and python versions, and
    independent between cohorts so adding a cohort never perturbs the others."""
    tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    return np.random.default_rng([master_seed, tag])


def participant_indices(cohort, n_boot, rng):
    """Class-stratified cluster bootstrap over participants.

    Returns (flat_idx, indptr): replicate b is flat_idx[indptr[b]:indptr[b + 1]].
    """
    members = collections.defaultdict(list)  # participant -> row indices
    p_class = {}
    for r in cohort:
        members[r["participant_label"]].append(r["row_index"])
        p_class.setdefault(r["participant_label"], r["disease_label"])
        assert p_class[r["participant_label"]] == r["disease_label"], (
            f"participant {r['participant_label']} spans both classes"
        )

    parts = sorted(members)
    member_arr = [np.asarray(members[p], dtype=np.int32) for p in parts]
    by_class = collections.defaultdict(list)
    for j, p in enumerate(parts):
        by_class[p_class[p]].append(j)
    strata = [(cls, np.asarray(sorted(js), dtype=np.int64)) for cls, js in sorted(by_class.items())]

    flat, indptr = [], [0]
    for _ in range(n_boot):
        picks = np.concatenate(
            [pool[rng.integers(0, pool.size, pool.size)] for _cls, pool in strata]
        )
        rep = np.concatenate([member_arr[j] for j in picks])
        flat.append(rep)
        indptr.append(indptr[-1] + rep.size)
    return np.concatenate(flat).astype(np.int32), np.asarray(indptr, dtype=np.int64)


def specimen_indices(cohort, n_boot, rng):
    """Class-stratified specimen-level bootstrap. Shape (n_boot, n_rows)."""
    by_class = collections.defaultdict(list)
    for r in cohort:
        by_class[r["disease_label"]].append(r["row_index"])
    strata = [np.asarray(sorted(v), dtype=np.int32) for _k, v in sorted(by_class.items())]
    cols = [rng.integers(0, pool.size, size=(n_boot, pool.size)) for pool in strata]
    return np.concatenate([pool[c] for pool, c in zip(strata, cols)], axis=1).astype(np.int32)


def write_cohort(path, cohort):
    """Write a canonical cohort table (list of dicts keyed by COHORT_FIELDS)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COHORT_FIELDS)
        w.writeheader()
        w.writerows(cohort)


def load_cohort(index_dir, name):
    """Read ``<index_dir>/cohorts/<name>.csv``.

    Returns (rows, y, id_to_row): the table's rows as dicts, the 0/1 labels in
    row order, and normalized specimen id -> row index.
    """
    with open(os.path.join(index_dir, "cohorts", f"{name}.csv")) as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([int(r["disease_label"]) for r in rows], dtype=np.int8)
    id_to_row = {r["specimen_id"]: int(r["row_index"]) for r in rows}
    return rows, y, id_to_row


def load_indices(index_dir, name, scheme):
    """Read ``<index_dir>/indices/<name>__<scheme>.npz`` as replicate multiplicities.

    Returns counts with counts[b, i] = number of times cohort row i is drawn in
    replicate b, which is what the *_from_counts metrics take.
    """
    z = np.load(os.path.join(index_dir, "indices", f"{name}__{scheme}.npz"), allow_pickle=False)
    n_boot, n_rows = int(z["n_boot"]), int(z["n_rows"])
    counts = np.zeros((n_boot, n_rows), dtype=np.float32)
    if scheme == "participant":
        flat, indptr = z["flat_idx"], z["indptr"]
        rows = np.repeat(np.arange(n_boot, dtype=np.int64), np.diff(indptr))
        np.add.at(counts, (rows, flat.astype(np.int64)), 1.0)
    else:
        idx = z["idx"].astype(np.int64)
        np.add.at(counts, (np.repeat(np.arange(n_boot), idx.shape[1]), idx.ravel()), 1.0)
    return counts


def auroc_from_counts(y, scores, counts):
    """Exact tie-aware AUROC for every replicate at once.

    counts[b, i] = multiplicity of cohort row i in replicate b. Uses the
    Mann-Whitney U identity summed over tie groups of the score, which is exact
    under duplication and O(n) per replicate.
    """
    perm = np.argsort(scores, kind="stable")
    s_sorted = scores[perm]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    cs = counts[:, perm]
    is_pos = (y[perm] == 1).astype(np.float32)
    pos_g = np.add.reduceat(cs * is_pos, starts, axis=1)
    neg_g = np.add.reduceat(cs * (1.0 - is_pos), starts, axis=1)
    cum_neg_below = np.cumsum(neg_g, axis=1) - neg_g
    u = (pos_g * (cum_neg_below + 0.5 * neg_g)).sum(axis=1)
    n1, n0 = pos_g.sum(axis=1), neg_g.sum(axis=1)
    return u / (n1 * n0)


def auprc_from_counts(y, scores, counts):
    """Average precision for every replicate at once.

    Step-wise AP, sum_k (R_k - R_{k-1}) * P_k over thresholds taken in order of
    decreasing score, with tied scores collapsed into a single threshold. This is
    the definition sklearn.metrics.average_precision_score uses -- not the
    trapezoidal PR-curve integral, which is optimistically biased.
    """
    perm = np.argsort(-scores, kind="stable")
    s_sorted = scores[perm]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    cs = counts[:, perm].astype(np.float64)
    is_pos = (y[perm] == 1).astype(np.float64)
    pos_g = np.add.reduceat(cs * is_pos, starts, axis=1)
    neg_g = np.add.reduceat(cs * (1.0 - is_pos), starts, axis=1)
    tp = np.cumsum(pos_g, axis=1)
    fp = np.cumsum(neg_g, axis=1)
    # Rows not drawn in a replicate still occupy tie groups, so leading groups can
    # have cumulative tp+fp == 0. Their recall increment is 0, so the term must
    # contribute 0 -- but 0/0 would give nan and 0*nan would poison the sum.
    denom = tp + fp
    prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    rec = tp / tp[:, -1:]
    d_rec = np.diff(rec, axis=1, prepend=0.0)
    return (d_rec * prec).sum(axis=1)


def point_auroc(y, scores):
    c = np.ones((1, y.size), dtype=np.float32)
    return float(auroc_from_counts(y, scores, c)[0])


def point_auprc(y, scores):
    c = np.ones((1, y.size), dtype=np.float32)
    return float(auprc_from_counts(y, scores, c)[0])
