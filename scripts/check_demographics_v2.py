"""
Manuscript-quality demographic visualizations for TCR cohort data.

Generates polished figures showing distributions of age, sex, and ancestry
across disease classes.

- Age: violin + strip plot
- Sex: 100% stacked horizontal bar
- Ancestry: 100% stacked horizontal bar

Input: Mal-ID metadata.tsv
Output: PNG figures in output_check_demographics_v2/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
METADATA_PATH = Path("/Users/lielcl/Library/CloudStorage/Dropbox/PyCharm/Mal-ID/data/metadata.tsv")
OUTPUT_DIR = Path(__file__).parent / "output_check_demographics_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
DISEASE_ORDER = ["Healthy/Background", "HIV", "Lupus", "Covid19", "T1D", "Influenza"]
DISEASE_SHORT = {
    "Healthy/Background": "Healthy",
    "HIV": "HIV",
    "Lupus": "Lupus",
    "Covid19": "COVID-19",
    "T1D": "T1D",
    "Influenza": "Influenza",
}
SEX_ORDER = ["M", "F", "Unknown"]
ANCESTRY_ORDER = ["Caucasian", "African", "Asian", "Hispanic/Latino", "Unknown"]

# ── Color palettes ───────────────────────────────────────────────────────────
# Disease colors (colorblind-friendly, from manuscript palette)
DISEASE_COLORS = {
    "Healthy/Background": "#999999",  # Grey
    "HIV":                "#028E68",  # Bluish Green
    "Lupus":              "#CC79A7",  # Reddish Purple
    "Covid19":            "#56B4E9",  # Sky Blue
    "T1D":                "#E69F00",  # Orange
    "Influenza":          "#F0E442",  # Yellow
}

# Sex colors: blue/pink tones
SEX_COLORS = {
    "M":       "#3A7CA5",  # Steel blue
    "F":       "#D1495B",  # Rose
    "Unknown": "#AAAAAA",  # Grey
}

# Ancestry colors: earth/warm tones (distinct from sex palette)
ANCESTRY_COLORS = {
    "Caucasian":       "#E8A838",  # Amber
    "African":         "#6A4C93",  # Purple
    "Asian":           "#1B998B",  # Teal
    "Hispanic/Latino": "#C1440E",  # Rust
    "Unknown":         "#AAAAAA",  # Grey
}

# ── Shared figure style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})
SAVE_DPI = 600


def load_tcr_metadata() -> pd.DataFrame:
    """Load metadata and filter to participants with TCR locus data."""
    df = pd.read_csv(METADATA_PATH, sep="\t")
    print(f"Total metadata rows: {len(df)}")

    # Keep only rows with TCR data
    has_tcr = df["available_gene_loci"].str.contains("TCR", na=False)
    df_tcr = df[has_tcr].copy()
    print(f"Rows with TCR data:  {len(df_tcr)}")

    # Validate all expected diseases are present
    found_diseases = set(df_tcr["disease"].unique())
    expected = set(DISEASE_ORDER)
    missing = expected - found_diseases
    if missing:
        raise ValueError(f"Expected diseases not found in TCR data: {missing}")

    # Keep only expected diseases (drop any unexpected ones)
    unexpected = found_diseases - expected
    if unexpected:
        print(f"  Dropping unexpected diseases: {unexpected}")
        df_tcr = df_tcr[df_tcr["disease"].isin(DISEASE_ORDER)].copy()

    # Normalize missing values to "Unknown"
    df_tcr["sex"] = df_tcr["sex"].fillna("Unknown")
    df_tcr["ancestry"] = df_tcr["ancestry"].fillna("Unknown")
    n_age_missing = df_tcr["age"].isna().sum()
    print(f"  Missing age values: {n_age_missing}")

    # Validate sex and ancestry values
    unexpected_sex = set(df_tcr["sex"].unique()) - set(SEX_ORDER)
    if unexpected_sex:
        raise ValueError(f"Unexpected sex values: {unexpected_sex}")
    unexpected_ancestry = set(df_tcr["ancestry"].unique()) - set(ANCESTRY_ORDER)
    if unexpected_ancestry:
        raise ValueError(f"Unexpected ancestry values: {unexpected_ancestry}")

    # Print per-disease summary
    print("\nPer-disease summary (TCR only):")
    for disease in DISEASE_ORDER:
        sub = df_tcr[df_tcr["disease"] == disease]
        n = len(sub)
        n_age_na = sub["age"].isna().sum()
        n_sex_na = (sub["sex"] == "Unknown").sum()
        n_anc_na = (sub["ancestry"] == "Unknown").sum()
        short = DISEASE_SHORT[disease]
        print(f"  {short:12s}  n={n:3d}  age_unknown={n_age_na:2d}  sex_unknown={n_sex_na:2d}  ancestry_unknown={n_anc_na:2d}")

    return df_tcr


# ═════════════════════════════════════════════════════════════════════════════
# AGE: VIOLIN + STRIP PLOT
# ═════════════════════════════════════════════════════════════════════════════

def plot_age_violin(df: pd.DataFrame) -> None:
    """Violin + strip plot of age by disease. Whiskers show Q1-Q3 (not
    min-max). Disease label and n= annotation are on separate lines below
    the axis, with n= in smaller font."""
    fig, ax = plt.subplots(figsize=(7, 5))

    positions = list(range(len(DISEASE_ORDER)))
    quartile_hw = 0.12  # half-width of Q1/Q3 horizontal caps
    median_hw = 0.16    # half-width of median horizontal line

    for i, disease in enumerate(DISEASE_ORDER):
        ages = df.loc[df["disease"] == disease, "age"].dropna().values
        if len(ages) < 2:
            continue
        color = DISEASE_COLORS[disease]

        # Violin body only (no built-in whiskers or median)
        vp = ax.violinplot(ages, positions=[i], showmedians=False,
                           showextrema=False, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.4)

        # Draw Q1-median-Q3 whiskers in black
        q1, median, q3 = np.percentile(ages, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", linewidth=1.2)
        ax.hlines(median, i - median_hw, i + median_hw,
                  color="black", linewidth=1.2)
        ax.hlines([q1, q3], i - quartile_hw, i + quartile_hw,
                  color="black", linewidth=1.2)

        # Strip (jittered dots) -- darken T1D and Influenza for visibility
        dot_color = color
        if disease in ("T1D", "Influenza"):
            from matplotlib.colors import to_rgb
            r, g, b = to_rgb(color)
            dot_color = (r * 0.75, g * 0.75, b * 0.75)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(ages))
        ax.scatter(i + jitter, ages, s=8, alpha=0.5, color=dot_color,
                   zorder=3, edgecolors="none")

    # Disease name tick labels (normal size)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISEASE_SHORT[d] for d in DISEASE_ORDER], fontsize=13,
                       rotation=45, ha="center")

    # Pale dotted horizontal gridlines at major y ticks
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, color="#BBBBBB", alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_ylabel("Age (years)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13, length=0)

    # n= annotation above each violin, above the 100 line
    for i, disease in enumerate(DISEASE_ORDER):
        sub = df[df["disease"] == disease]
        n_total = len(sub)
        n_unknown = sub["age"].isna().sum()
        label = f"n={n_total}"
        if n_unknown > 0:
            label += f"\n({n_unknown} unk.)"
        ax.text(i, 102, label, ha="center", va="bottom",
                fontsize=10, color="#999999")

    ax.set_ylim(ax.get_ylim()[0], 102)

    plt.subplots_adjust(bottom=0.18)
    path = OUTPUT_DIR / "age_violin.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SEX: 100% STACKED HORIZONTAL BAR
# ═════════════════════════════════════════════════════════════════════════════

def _build_sex_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Build a disease x sex count table."""
    counts = (
        df.groupby(["disease", "sex"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=DISEASE_ORDER, columns=SEX_ORDER, fill_value=0)
    )
    counts.index = [DISEASE_SHORT[d] for d in counts.index]
    return counts


def plot_sex_stacked_pct(df: pd.DataFrame) -> None:
    """100% stacked horizontal bar chart of sex proportions, with legend
    placed to the right of the figure."""
    counts = _build_sex_counts(df)
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    y = np.arange(len(pcts))
    left = np.zeros(len(pcts))
    bar_height = 0.5

    for sex in SEX_ORDER:
        vals = pcts[sex].values
        ax.barh(y, vals, left=left, height=bar_height,
                label=sex, color=SEX_COLORS[sex], edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val >= 10:
                ax.text(left[i] + val / 2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pcts.index, fontsize=11)
    ax.set_xlabel("Percentage", fontsize=11)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    ax.legend(frameon=False, fontsize=10.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3)

    plt.tight_layout()
    path = OUTPUT_DIR / "sex_stacked_pct.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ANCESTRY: 100% STACKED HORIZONTAL BAR
# ═════════════════════════════════════════════════════════════════════════════

def _build_ancestry_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Build a disease x ancestry count table."""
    counts = (
        df.groupby(["disease", "ancestry"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=DISEASE_ORDER, columns=ANCESTRY_ORDER, fill_value=0)
    )
    counts.index = [DISEASE_SHORT[d] for d in counts.index]
    return counts


def plot_ancestry_stacked_pct(df: pd.DataFrame) -> None:
    """100% stacked horizontal bar chart of ancestry proportions, with legend
    placed to the right of the figure."""
    counts = _build_ancestry_counts(df)
    totals = counts.sum(axis=1)
    pcts = counts.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    y = np.arange(len(pcts))
    left = np.zeros(len(pcts))
    bar_height = 0.5

    for anc in ANCESTRY_ORDER:
        vals = pcts[anc].values
        ax.barh(y, vals, left=left, height=bar_height,
                label=anc, color=ANCESTRY_COLORS[anc], edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val >= 8:
                ax.text(left[i] + val / 2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pcts.index, fontsize=11)
    ax.set_xlabel("Percentage", fontsize=11)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)

    ax.legend(frameon=False, fontsize=10.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)

    plt.tight_layout()
    path = OUTPUT_DIR / "ancestry_stacked_pct.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table and save as CSV."""
    rows = []
    for disease in DISEASE_ORDER:
        sub = df[df["disease"] == disease]
        n = len(sub)
        ages = sub["age"].dropna()
        age_str = f"{ages.median():.0f} ({ages.min():.0f}-{ages.max():.0f})" if len(ages) > 0 else "N/A"
        n_age_unk = sub["age"].isna().sum()
        n_male = (sub["sex"] == "M").sum()
        n_female = (sub["sex"] == "F").sum()
        n_sex_unk = (sub["sex"] == "Unknown").sum()
        ancestry_counts = sub["ancestry"].value_counts()
        anc_parts = []
        for anc in ANCESTRY_ORDER:
            if anc in ancestry_counts and ancestry_counts[anc] > 0:
                anc_parts.append(f"{anc}: {ancestry_counts[anc]}")
        rows.append({
            "Disease": DISEASE_SHORT[disease],
            "N": n,
            "Age median (range)": age_str,
            "Age unknown": n_age_unk,
            "Male": n_male,
            "Female": n_female,
            "Sex unknown": n_sex_unk,
            "Ancestry": "; ".join(anc_parts),
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("DEMOGRAPHICS SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    csv_path = OUTPUT_DIR / "demographics_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# COMBINED PANEL (all 3 side by side)
# ═════════════════════════════════════════════════════════════════════════════

def plot_combined_panel(df: pd.DataFrame) -> None:
    """Three-panel figure: (a) age violin, (b) sex stacked %, (c) ancestry
    stacked %. Designed to be a single manuscript figure."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(13, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.35)

    diseases_short = [DISEASE_SHORT[d] for d in DISEASE_ORDER]
    positions = list(range(len(DISEASE_ORDER)))
    quartile_hw = 0.12
    median_hw = 0.16

    # ── Panel A: Age violin ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])

    for i, disease in enumerate(DISEASE_ORDER):
        ages = df.loc[df["disease"] == disease, "age"].dropna().values
        if len(ages) < 2:
            continue
        color = DISEASE_COLORS[disease]

        vp = ax.violinplot(ages, positions=[i], showmedians=False,
                           showextrema=False, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.4)

        q1, median, q3 = np.percentile(ages, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", linewidth=1.2)
        ax.hlines(median, i - median_hw, i + median_hw,
                  color="black", linewidth=1.2)
        ax.hlines([q1, q3], i - quartile_hw, i + quartile_hw,
                  color="black", linewidth=1.2)

        dot_color = color
        if disease in ("T1D", "Influenza"):
            from matplotlib.colors import to_rgb
            r, g, b = to_rgb(color)
            dot_color = (r * 0.75, g * 0.75, b * 0.75)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(ages))
        ax.scatter(i + jitter, ages, s=8, alpha=0.5, color=dot_color,
                   zorder=3, edgecolors="none")

    ax.yaxis.grid(True, linestyle=":", linewidth=0.7, color="#BBBBBB", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(positions)
    ax.set_xticklabels(diseases_short, fontsize=11, rotation=45, ha="center")
    ax.set_ylabel("Age (years)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11, length=0)

    # n= annotations above violins
    for i, disease in enumerate(DISEASE_ORDER):
        sub = df[df["disease"] == disease]
        n_total = len(sub)
        n_unknown = sub["age"].isna().sum()
        label = f"n={n_total}"
        if n_unknown > 0:
            label += f"\n({n_unknown} unk.)"
        ax.text(i, 102, label, ha="center", va="bottom",
                fontsize=8.5, color="#999999")

    ax.set_ylim(ax.get_ylim()[0], 102)

    # ── Panel B: Sex stacked % ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    sex_counts = _build_sex_counts(df)
    totals = sex_counts.sum(axis=1)
    pcts = sex_counts.div(totals, axis=0) * 100

    y = np.arange(len(pcts))
    left = np.zeros(len(pcts))
    for sex in SEX_ORDER:
        vals = pcts[sex].values
        ax.barh(y, vals, left=left, height=0.5,
                label=sex, color=SEX_COLORS[sex], edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val >= 10:
                ax.text(left[i] + val / 2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pcts.index, fontsize=10)
    ax.set_xlabel("Percentage", fontsize=11)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3)

    # ── Panel C: Ancestry stacked % ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    anc_counts = _build_ancestry_counts(df)
    totals = anc_counts.sum(axis=1)
    pcts = anc_counts.div(totals, axis=0) * 100

    y = np.arange(len(pcts))
    left = np.zeros(len(pcts))
    for anc in ANCESTRY_ORDER:
        vals = pcts[anc].values
        ax.barh(y, vals, left=left, height=0.5,
                label=anc, color=ANCESTRY_COLORS[anc], edgecolor="white", linewidth=0.5)
        for i, val in enumerate(vals):
            if val >= 8:
                ax.text(left[i] + val / 2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pcts.index, fontsize=10)
    ax.set_xlabel("Percentage", fontsize=11)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3,
              columnspacing=0.8, handlelength=1.0, handletextpad=0.3)

    path = OUTPUT_DIR / "combined_panel.png"
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    df = load_tcr_metadata()
    print(f"\nGenerating figures in: {OUTPUT_DIR}\n")

    plot_age_violin(df)
    plot_sex_stacked_pct(df)
    plot_ancestry_stacked_pct(df)
    plot_combined_panel(df)
    print_summary_table(df)

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
