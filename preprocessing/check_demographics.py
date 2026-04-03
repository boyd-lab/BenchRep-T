import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

METADATA_PATH = "/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/data/malid_clean/metadata.tsv"
TCR_DIR = "/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/data/malid_clean/TCR"
OUT_DIR = "/oak/stanford/groups/akundaje/abuen/tcr-bench/airr_bench/preprocessing/demographics_plots"
DISEASE_CLASSES = ["Healthy/Background", "HIV", "Lupus", "Covid19", "T1D", "Influenza"]

os.makedirs(OUT_DIR, exist_ok=True)

# Parse TCR directory: filenames are part_table_{participant}_{specimen}.tsv.gz
tcr_keys = set()
pattern = re.compile(r"^part_table_(.+)\.tsv\.gz$")
for fname in os.listdir(TCR_DIR):
    m = pattern.match(fname)
    if m:
        tcr_keys.add(m.group(1))  # "BFI-XXXX_SPECIMEN"

# Load metadata
df = pd.read_csv(METADATA_PATH, sep="\t")
df["tcr_key"] = df["participant_label"] + "_" + df["specimen_label"]
df["has_tcr"] = df["tcr_key"].isin(tcr_keys)

print(f"Total metadata rows: {len(df)}")
print(f"Rows with TCR data:  {df['has_tcr'].sum()}")
print()

df_tcr = df[df["has_tcr"]].copy()

# Normalize sex and ancestry
df_tcr["sex"] = df_tcr["sex"].fillna("Unknown")
df_tcr["ancestry"] = df_tcr["ancestry"].fillna("Unknown")

print("Per-disease TCR availability:")
for disease in DISEASE_CLASSES:
    total = (df["disease"] == disease).sum()
    with_tcr = (df_tcr["disease"] == disease).sum()
    print(f"  {disease}: {with_tcr}/{total} have TCR data")
print()

# ── Plotting ──────────────────────────────────────────────────────────────────

SEX_COLORS = {"M": "#4C72B0", "F": "#DD8452", "Unknown": "#8c8c8c"}
RACE_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]

fig, axes = plt.subplots(
    nrows=len(DISEASE_CLASSES),
    ncols=3,
    figsize=(16, 4 * len(DISEASE_CLASSES)),
)
fig.suptitle("Demographics of patients with TCR data", fontsize=16, fontweight="bold", y=1.01)

for row_idx, disease in enumerate(DISEASE_CLASSES):
    sub = df_tcr[df_tcr["disease"] == disease]
    ax_sex, ax_age, ax_race = axes[row_idx]

    # ── Sex bar chart ──────────────────────────────────────────────────────
    sex_counts = sub["sex"].value_counts().reindex(["M", "F", "Unknown"], fill_value=0)
    colors = [SEX_COLORS[s] for s in sex_counts.index]
    bars = ax_sex.bar(sex_counts.index, sex_counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, sex_counts.values):
        if val > 0:
            ax_sex.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                        ha="center", va="bottom", fontsize=9)
    ax_sex.set_title(f"{disease}\nSex", fontsize=11)
    ax_sex.set_ylabel("Count")
    ax_sex.set_xlabel("")
    ax_sex.spines[["top", "right"]].set_visible(False)
    ax_sex.set_ylim(0, sex_counts.max() * 1.2 + 1)

    # ── Age histogram ──────────────────────────────────────────────────────
    ages = sub["age"].dropna()
    if len(ages) > 0:
        bins = range(int(ages.min()), int(ages.max()) + 5, 5)
        ax_age.hist(ages, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.8)
        ax_age.axvline(ages.median(), color="#C44E52", linestyle="--", linewidth=1.5, label=f"Median: {ages.median():.0f}")
        ax_age.legend(fontsize=8)
    ax_age.set_title(f"{disease}\nAge Distribution", fontsize=11)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Count")
    ax_age.spines[["top", "right"]].set_visible(False)

    # ── Race/ancestry bar chart ────────────────────────────────────────────
    race_counts = sub["ancestry"].value_counts()
    race_colors = RACE_PALETTE[: len(race_counts)]
    bars = ax_race.bar(range(len(race_counts)), race_counts.values, color=race_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, race_counts.values):
        ax_race.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                     ha="center", va="bottom", fontsize=9)
    ax_race.set_xticks(range(len(race_counts)))
    ax_race.set_xticklabels(race_counts.index, rotation=30, ha="right", fontsize=8)
    ax_race.set_title(f"{disease}\nRace/Ancestry", fontsize=11)
    ax_race.set_ylabel("Count")
    ax_race.spines[["top", "right"]].set_visible(False)
    ax_race.set_ylim(0, race_counts.max() * 1.2 + 1)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "demographics_by_disease.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
