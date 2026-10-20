"""
Pre-assign bootstrap resample indices for the demographic-confounding table.

Same design as ``evals.bootstrap_indices_disease_classification`` -- B=10,000,
class-stratified participant-cluster bootstrap, n-out-of-n -- but over the
demographic-confounding cohorts, which are a *different* sample set from the
disease-classification cohorts: each is the complete-demographics subset the
feature_concat analysis was actually run on (Lupus 229, HIV 253, Covid19 211,
CMV 542), not the full disease cohort.

One index set per disease, shared by all seven method/setting cells, so every
cell's SE lands on identical resamples and any cell-vs-cell difference can be
bootstrapped by subtracting per-replicate vectors (paired bootstrap).

Cohort definition: the intersection of specimens present in all seven files.
The six TCR files agree exactly within a disease; demographics_only covers a
superset for Lupus/HIV/Covid19, so the intersection equals the TCR cohort and
demographics_only gets restricted onto it. Left unrestricted, the
demographics-only cell would sit on both a different sample set and a different
class prevalence (Lupus 95/287 = 33.1% vs 63/229 = 27.5%), and AUPRC is
directly sensitive to prevalence.

Seeds are drawn from "democonfound:<disease>", so these streams are independent
of the same-named disease-classification cohorts.

--probabilities_dir holds <disease>/<stem>.csv for every stem in CELLS. Writes
cohorts/<disease>.csv and indices/<disease>__{participant,specimen}.npz under
--output_dir, consumed by ``evals.bootstrap_stats_demographic_confounding``.
"""

import argparse
import csv
import os

import numpy as np

from utils.bootstrap import (
    N_BOOT,
    cohort_rng,
    normalize_id,
    participant_indices,
    specimen_indices,
    write_cohort,
)


DISEASES = ["Lupus", "HIV", "Covid19", "CMV"]

# (label, filename stem). The cohort is the intersection over all of these.
CELLS = [
    ("Demographics only", "demographics_only"),
    ("VJ (Reg)", "VJ__repertoire_only"),
    ("VJ (Reg)", "VJ__repertoire_plus_demographics"),
    ("DeepTCR", "DeepTCR__repertoire_only"),
    ("DeepTCR", "DeepTCR__repertoire_plus_demographics"),
    ("ABMIL", "ABMIL__repertoire_only"),
    ("ABMIL", "ABMIL__repertoire_plus_demographics"),
]


def read_cells(prob_dir, disease):
    """Read all seven probability files for one disease."""
    out = {}
    for _label, stem in CELLS:
        path = os.path.join(prob_dir, disease, f"{stem}.csv")
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        seen = {}
        for r in rows:
            sid = normalize_id(r["specimen_label"])
            if sid in seen:
                raise ValueError(f"{disease}/{stem}: duplicate specimen {sid}")
            seen[sid] = r
        out[stem] = seen
    return out


def build_cohort(disease, cells):
    """Canonical sample table: the specimens every cell covers, in file order."""
    common = set.intersection(*(set(v) for v in cells.values()))
    # Order by the repertoire-only VJ file, which is the experiment's base cohort.
    base = cells["VJ__repertoire_only"]
    ordered = [sid for sid in base if sid in common]
    if len(ordered) != len(common):
        raise ValueError(f"{disease}: base file misses {len(common) - len(ordered)} common ids")

    cohort = []
    for i, sid in enumerate(ordered):
        r = base[sid]
        row = {
            "row_index": i,
            "specimen_id": sid,
            "specimen_id_as_written": r["specimen_label"],
            "participant_label": r["participant_label"],
            "disease_label": int(r["disease_label"]),
            "disease_label_str": r["disease_label_str"],
            "fold": int(r["malid_cross_validation_fold_id_when_in_test_set"]),
        }
        # Every cell must agree on the label for this specimen.
        for stem, tbl in cells.items():
            if int(tbl[sid]["disease_label"]) != row["disease_label"]:
                raise ValueError(f"{disease}/{stem}: label disagreement at {sid}")
        cohort.append(row)
    return cohort


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--probabilities_dir", required=True,
                    help="Directory with one <disease>/ subdirectory of cell CSVs.")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write cohorts/ and indices/.")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output_dir, "cohorts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "indices"), exist_ok=True)

    for disease in DISEASES:
        cells = read_cells(args.probabilities_dir, disease)
        cohort = build_cohort(disease, cells)
        n_rows = len(cohort)
        n_pos = sum(r["disease_label"] for r in cohort)
        n_part = len({r["participant_label"] for r in cohort})
        dropped = {
            stem: len(tbl) - n_rows for stem, tbl in cells.items() if len(tbl) != n_rows
        }

        write_cohort(os.path.join(args.output_dir, "cohorts", f"{disease}.csv"), cohort)

        # One stream for both schemes: the specimen draws continue where the
        # participant draws stop, as in the published index sets.
        rng = cohort_rng(f"democonfound:{disease}")
        flat, indptr = participant_indices(cohort, N_BOOT, rng)
        np.savez_compressed(
            os.path.join(args.output_dir, "indices", f"{disease}__participant.npz"),
            flat_idx=flat, indptr=indptr,
            n_boot=np.int64(N_BOOT), n_rows=np.int64(n_rows),
        )
        idx = specimen_indices(cohort, N_BOOT, rng)
        np.savez_compressed(
            os.path.join(args.output_dir, "indices", f"{disease}__specimen.npz"),
            idx=idx, n_boot=np.int64(N_BOOT), n_rows=np.int64(n_rows),
        )

        print(f"{disease:8s} n={n_rows:4d}  pos={n_pos:4d}  participants={n_part:4d}"
              f"  prevalence={n_pos / n_rows:.3f}")
        for stem, extra in sorted(dropped.items()):
            print(f"           restricted {stem}: dropped {extra} rows outside the common cohort")

    print(f"\nWrote cohorts + indices to {args.output_dir}")


if __name__ == "__main__":
    main()
