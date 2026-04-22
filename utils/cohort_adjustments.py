"""
Per-disease cohort adjustments for fair disease-vs-healthy comparison.

Three modes are supported:

    'filter'
        Symmetric demographic filter applied to both disease and healthy
        rows (e.g. HIV -> ancestry == 'African').

    'age_match_healthy'
        Leave the disease cohort unchanged and subsample Healthy/Background
        so its age histogram (bin_width-year bins) has the same shape as
        the disease cohort's.

    'random_baseline' (control)
        Requested via ``random_baseline=True`` in combination with one of
        the two modes above. The disease side is identical to the
        demographic-matched run. The healthy side is replaced with a
        uniformly-random sample of the same target N (ignoring
        demographics). Used to isolate covariate effects from the
        cohort-size effect alone.
"""

import numpy as np
import pandas as pd


DEMOGRAPHIC_ADJUSTMENTS = {
    'HIV': {'mode': 'filter', 'ancestry': 'African'},
    'Lupus': {'mode': 'age_match_healthy', 'bin_width': 10},
    'T1D': {'mode': 'age_match_healthy', 'bin_width': 10},
    'Influenza': {'mode': 'age_match_healthy', 'bin_width': 10},
}


def apply_cohort_adjustment(df, target_disease, seed=7, random_baseline=False):
    """
    Dispatch to the adjustment strategy configured for ``target_disease``.

    Args:
        df: Combined disease+healthy DataFrame with a 'label' column
            (1 = disease, 0 = healthy).
        target_disease: Disease key in DEMOGRAPHIC_ADJUSTMENTS.
        seed: RNG seed for any sampling involved.
        random_baseline: If True, keep the disease cohort as the underlying
            rule would produce it, but resample healthy uniformly at random
            to the same target N.

    Returns:
        Adjusted DataFrame (subset of ``df``).
    """
    rule = DEMOGRAPHIC_ADJUSTMENTS.get(target_disease)
    if not rule:
        print(f"  No demographic adjustment defined for '{target_disease}' "
              f"- leaving cohort unchanged.")
        return df

    mode = rule.get('mode', 'filter')
    if mode == 'filter':
        return _apply_filter_adjustment(df, target_disease, rule,
                                        seed=seed,
                                        random_baseline=random_baseline)
    if mode == 'age_match_healthy':
        return _apply_age_match_adjustment(df, target_disease, rule,
                                           seed=seed,
                                           random_baseline=random_baseline)
    raise ValueError(f"Unknown cohort adjustment mode '{mode}' "
                     f"for disease '{target_disease}'")


def _apply_filter_adjustment(df, target_disease, rule, seed=7,
                             random_baseline=False):
    """Symmetric filter on both disease and healthy rows.

    When ``random_baseline`` is True, the disease side is still filtered
    (so the disease cohort matches the demographic-matched run), but the
    healthy side is replaced with a uniformly-random sample of size
    ``len(filtered_healthy)`` drawn from the *unfiltered* healthy pool.
    """
    mask = pd.Series(True, index=df.index)
    desc = []
    if 'ancestry' in rule:
        mask &= (df['ancestry'] == rule['ancestry'])
        desc.append(f"ancestry={rule['ancestry']}")
    if 'sex' in rule:
        mask &= (df['sex'] == rule['sex'])
        desc.append(f"sex={rule['sex']}")
    if 'age_min' in rule or 'age_max' in rule:
        age = pd.to_numeric(df['age'], errors='coerce')
        if 'age_min' in rule:
            mask &= (age >= rule['age_min'])
        if 'age_max' in rule:
            mask &= (age <= rule['age_max'])
        desc.append(f"age in [{rule.get('age_min', '-inf')},"
                    f"{rule.get('age_max', 'inf')}]")

    before = len(df)
    filtered = df[mask].copy()

    if not random_baseline:
        print(f"  Demographic filter for '{target_disease}' "
              f"({', '.join(desc)}): {before} -> {len(filtered)} rows")
        return filtered

    filtered_disease = filtered[filtered['label'] == 1]
    filtered_healthy = filtered[filtered['label'] == 0]
    all_healthy = df[df['label'] == 0]
    target_n = min(len(filtered_healthy), len(all_healthy))

    if target_n <= 0:
        print(f"  Warning: random baseline target N is 0 for "
              f"'{target_disease}'. Leaving cohort unchanged.")
        return df

    sampled_healthy = all_healthy.sample(n=target_n, random_state=int(seed))
    result = pd.concat([filtered_disease, sampled_healthy], axis=0)
    print(f"  Random-baseline cohort for '{target_disease}' "
          f"(disease filter: {', '.join(desc)}; healthy random): "
          f"disease {len(filtered_disease)} (filtered), "
          f"healthy {len(filtered_healthy)} target N "
          f"-> {len(sampled_healthy)} random sample from {len(all_healthy)} "
          f"(seed={seed})")
    return result


def _apply_age_match_adjustment(df, target_disease, rule, seed=7,
                                random_baseline=False):
    """Age-match healthy to disease's age histogram.

    When ``random_baseline`` is True, the same target N is computed from the
    age histogram, but the healthy subset is drawn uniformly at random from
    the full healthy pool (with no stratification by age).
    """
    bin_width = int(rule.get('bin_width', 10))

    disease_df = df[df['label'] == 1].copy()
    healthy_df = df[df['label'] == 0].copy()

    disease_age = pd.to_numeric(disease_df['age'], errors='coerce')
    healthy_age = pd.to_numeric(healthy_df['age'], errors='coerce')

    d_age_valid_mask = disease_age.notna()
    h_age_valid_mask = healthy_age.notna()

    n_disease_nan = int((~d_age_valid_mask).sum())
    n_healthy_nan = int((~h_age_valid_mask).sum())

    disease_with_age = disease_df[d_age_valid_mask]
    healthy_with_age = healthy_df[h_age_valid_mask]
    d_ages = disease_age[d_age_valid_mask]
    h_ages = healthy_age[h_age_valid_mask]

    if len(disease_with_age) == 0 or len(healthy_with_age) == 0:
        print(f"  Warning: insufficient age data to age-match "
              f"'{target_disease}'. Leaving cohort unchanged.")
        return df

    combined_min = float(min(d_ages.min(), h_ages.min()))
    combined_max = float(max(d_ages.max(), h_ages.max()))
    bin_start = int(np.floor(combined_min / bin_width)) * bin_width
    bin_end = (int(np.floor(combined_max / bin_width)) + 1) * bin_width
    bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)
    n_bins = len(bin_edges) - 1

    d_bin = pd.cut(d_ages, bin_edges, right=False, labels=False).astype(int)
    h_bin = pd.cut(h_ages, bin_edges, right=False, labels=False).astype(int)

    d_counts = np.bincount(d_bin.values, minlength=n_bins)
    h_counts = np.bincount(h_bin.values, minlength=n_bins)

    active_bins = [i for i in range(n_bins) if d_counts[i] > 0 and h_counts[i] > 0]
    uncovered_bins = [i for i in range(n_bins) if d_counts[i] > 0 and h_counts[i] == 0]

    if uncovered_bins:
        missed = int(sum(d_counts[i] for i in uncovered_bins))
        labels_str = ', '.join(f"[{bin_edges[i]},{bin_edges[i+1]})"
                               for i in uncovered_bins)
        print(f"  Warning: {missed} disease sample(s) fall in bin(s) "
              f"{labels_str} with no healthy counterparts; these bins are "
              f"excluded from the target distribution.")

    if not active_bins:
        print(f"  Warning: no coverable age bins for '{target_disease}'. "
              f"Leaving cohort unchanged.")
        return df

    active_d = np.array([d_counts[i] for i in active_bins], dtype=float)
    active_h = np.array([h_counts[i] for i in active_bins], dtype=float)
    props = active_d / active_d.sum()
    max_n = int(np.floor(float(np.min(active_h / props))))
    if max_n <= 0:
        print(f"  Warning: maximum matched healthy N is 0 for "
              f"'{target_disease}'. Leaving cohort unchanged.")
        return df

    if random_baseline:
        target_n = min(max_n, len(healthy_df))
        sampled_healthy = healthy_df.sample(n=target_n, random_state=int(seed))
        print(f"  Random-baseline cohort for '{target_disease}' "
              f"(target N from {bin_width}y age-match = {max_n}): "
              f"disease {len(disease_df)} (unchanged), "
              f"healthy {len(healthy_df)} -> {len(sampled_healthy)} "
              f"random sample (seed={seed})")
        return pd.concat([disease_df, sampled_healthy], axis=0)

    rng = np.random.RandomState(int(seed))
    sampled_parts = []
    h_final_counts = np.zeros(n_bins, dtype=int)
    h_bin_by_index = pd.Series(h_bin.values, index=healthy_with_age.index)

    for j, bin_idx in enumerate(active_bins):
        target = int(round(max_n * props[j]))
        target = min(target, int(active_h[j]))
        if target <= 0:
            continue
        pool = healthy_with_age.loc[h_bin_by_index[h_bin_by_index == bin_idx].index]
        sampled = pool.sample(n=target, random_state=int(rng.randint(0, 2**31 - 1)))
        sampled_parts.append(sampled)
        h_final_counts[bin_idx] = target

    sampled_healthy = (pd.concat(sampled_parts, axis=0)
                       if sampled_parts else healthy_with_age.iloc[0:0])

    print(f"  Age-matched cohort for '{target_disease}' "
          f"(bin width {bin_width}y): disease {len(disease_df)} (unchanged), "
          f"healthy {len(healthy_df)} -> {len(sampled_healthy)}")
    print(f"    Per-bin (disease / healthy available / healthy sampled):")
    for i in range(n_bins):
        if d_counts[i] == 0 and h_counts[i] == 0:
            continue
        print(f"      [{bin_edges[i]:3d},{bin_edges[i+1]:3d}):  "
              f"{d_counts[i]:3d}  /  {h_counts[i]:3d}  /  {h_final_counts[i]:3d}")
    if n_disease_nan:
        print(f"    Note: {n_disease_nan} disease row(s) have missing age "
              f"(kept, not used for histogram)")
    if n_healthy_nan:
        print(f"    Note: {n_healthy_nan} healthy row(s) dropped for missing age")

    return pd.concat([disease_df, sampled_healthy], axis=0)
