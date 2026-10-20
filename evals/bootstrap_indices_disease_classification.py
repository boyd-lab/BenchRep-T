"""
Pre-assign bootstrap resample indices for the disease-classification table.

One index set per disease cohort, shared by every method, so all per-cell CIs
and all method-vs-method difference tests are computed on identical resamples
(paired bootstrap). Writes, under --output_dir:

    cohorts/<cohort>.csv                 canonical sample table; indices point
                                         into its row_index column
    indices/<cohort>__participant.npz    participant-cluster scheme (primary)
    indices/<cohort>__specimen.npz       specimen-level scheme (sensitivity)
    index_manifest.csv                   per-cohort row and donor counts

Design and seeding are in ``utils.bootstrap``. Generation is deterministic, so
rerunning on the same probability files reproduces the published index sets
exactly. Consumed by ``evals.bootstrap_stats_disease_classification`` and
``evals.bootstrap_stats_abmil_encoder_sweep``.

--probabilities_dir holds one pooled out-of-fold prediction CSV per method and
disease (see ``evals.bootstrap_stats_disease_classification`` for the naming).
The canonical table for each disease is taken from the ABMIL file, which carries
participant_label and covers every repertoire.
"""

import argparse
import csv
import os

import numpy as np

from utils.bootstrap import (
    MASTER_SEED,
    N_BOOT,
    cohort_rng,
    normalize_id,
    participant_indices,
    specimen_indices,
    write_cohort,
)


CANONICAL_SOURCE = {
    "Covid19": "ABMIL__Covid19.csv",
    "HIV": "ABMIL__HIV.csv",
    "Influenza": "ABMIL__Influenza.csv",
    "Lupus": "ABMIL__Lupus.csv",
    "T1D": "ABMIL__T1D.csv",
    "RA": "ABMIL__Rheumatoid_Arthritis.csv",
    "Tb": "ABMIL__TB_progression.csv",
}
# A method that covers only part of a cohort gets its own index set over the
# subset it does cover, so its CI is valid and differences against it are
# computed on a cohort both methods share.
#   Mal-ID RA: 90 of 91 (missing RA6-SF)
SUBSET_COHORTS = {
    "RA_malid90": ("RA", "Mal-ID_RA.csv"),
}
COHORTS = ["Covid19", "HIV", "Influenza", "Lupus", "T1D", "RA", "RA_malid90", "Tb"]


def read_prob_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    id_col = "specimen_label" if "specimen_label" in rows[0] else "sample_name"
    fold_col = (
        "malid_cross_validation_fold_id_when_in_test_set"
        if "malid_cross_validation_fold_id_when_in_test_set" in rows[0]
        else "fold"
    )
    return rows, id_col, fold_col


def build_cohort(prob_dir, disease):
    """Canonical sample table for one disease: one row per repertoire."""
    rows, id_col, fold_col = read_prob_csv(os.path.join(prob_dir, CANONICAL_SOURCE[disease]))
    out = []
    for i, x in enumerate(rows):
        out.append(
            {
                "row_index": i,
                "specimen_id": normalize_id(x[id_col]),
                "specimen_id_as_written": x[id_col],
                "participant_label": x["participant_label"],
                "disease_label": int(x["disease_label"]),
                "disease_label_str": x["disease_label_str"],
                "fold": int(x[fold_col]),
            }
        )
    return out


def subset_cohort(cohort, keep_ids):
    """Re-index a cohort down to keep_ids, preserving original order."""
    keep = {normalize_id(s) for s in keep_ids}
    out = [dict(r) for r in cohort if r["specimen_id"] in keep]
    for i, r in enumerate(out):
        r["row_index"] = i
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--probabilities_dir", required=True,
                    help="Directory of per-method, per-disease prediction CSVs.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write cohorts/, indices/ and index_manifest.csv.")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output_dir, "cohorts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "indices"), exist_ok=True)

    cohorts = {d: build_cohort(args.probabilities_dir, d) for d in CANONICAL_SOURCE}
    for sub_name, (parent, src) in SUBSET_COHORTS.items():
        rows, id_col, _ = read_prob_csv(os.path.join(args.probabilities_dir, src))
        cohorts[sub_name] = subset_cohort(cohorts[parent], [x[id_col] for x in rows])
        assert len(cohorts[sub_name]) == len(rows), f"{sub_name} subset join incomplete"

    manifest = []
    for name in COHORTS:
        cohort = cohorts[name]
        write_cohort(os.path.join(args.output_dir, "cohorts", f"{name}.csv"), cohort)

        n_rows = len(cohort)
        n_pos = sum(r["disease_label"] == 1 for r in cohort)
        parts = {r["participant_label"] for r in cohort}
        p_pos = len({r["participant_label"] for r in cohort if r["disease_label"] == 1})

        flat, indptr = participant_indices(cohort, N_BOOT, cohort_rng(name + "|participant"))
        np.savez_compressed(
            os.path.join(args.output_dir, "indices", f"{name}__participant.npz"),
            flat_idx=flat,
            indptr=indptr,
            n_boot=np.int64(N_BOOT),
            n_rows=np.int64(n_rows),
            master_seed=np.int64(MASTER_SEED),
            scheme=np.str_("participant-cluster, class-stratified, n-out-of-n"),
        )
        spec = specimen_indices(cohort, N_BOOT, cohort_rng(name + "|specimen"))
        np.savez_compressed(
            os.path.join(args.output_dir, "indices", f"{name}__specimen.npz"),
            idx=spec,
            n_boot=np.int64(N_BOOT),
            n_rows=np.int64(n_rows),
            master_seed=np.int64(MASTER_SEED),
            scheme=np.str_("specimen-level, class-stratified, n-out-of-n"),
        )

        lens = np.diff(indptr)
        manifest.append(
            {
                "cohort": name,
                "n_rows": n_rows,
                "n_pos": n_pos,
                "n_neg": n_rows - n_pos,
                "n_participants": len(parts),
                "n_participants_pos": p_pos,
                "n_participants_neg": len(parts) - p_pos,
                "n_boot": N_BOOT,
                "participant_rep_len_min": int(lens.min()),
                "participant_rep_len_median": int(np.median(lens)),
                "participant_rep_len_max": int(lens.max()),
                "specimen_rep_len": n_rows,
            }
        )
        print(
            f"{name:14} rows={n_rows:4} ({n_pos:3}+/{n_rows - n_pos:3}-)  "
            f"donors={len(parts):4} ({p_pos:3}+/{len(parts) - p_pos:3}-)  "
            f"replicate rows {lens.min()}-{lens.max()} (med {int(np.median(lens))})"
        )

    with open(os.path.join(args.output_dir, "index_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nWrote cohorts + indices to {args.output_dir}")


if __name__ == "__main__":
    main()
