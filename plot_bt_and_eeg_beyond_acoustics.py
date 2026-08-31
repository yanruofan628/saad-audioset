#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper-style figures: intrinsic priority + EEG beyond acoustics (Arial, large type)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BT = ROOT / "bt_intrinsic_priority"
PRIME = ROOT / "prime_ch4_supplement"
OUT = Path(r"A:\JBHI")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 24,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 1.8,
})

# Canonical AudioSet fine labels used in this experiment
CANON = [
    "Telephone bell ringing",
    "Ambulance (siren)",
    "Baby cry, infant cry",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Sad music",
    "Funny music",
    "Bass drum",
    "Computer keyboard",
    "Helicopter",
    "Chicken, rooster",
]
CANON_SORTED = sorted(CANON, key=len, reverse=True)

CAT_COLORS = {
    "Telephone bell ringing": "#C44E52",
    "Ambulance (siren)": "#E17C76",
    "Baby cry, infant cry": "#DD8452",
    "Male speech, man speaking": "#4C72B0",
    "Female speech, woman speaking": "#8172B3",
    "Child speech, kid speaking": "#A890C4",
    "Sad music": "#55A868",
    "Funny music": "#8BC34A",
    "Bass drum": "#64B5CD",
    "Computer keyboard": "#8C8C8C",
    "Helicopter": "#937860",
    "Chicken, rooster": "#CCB974",
}

SHORT = {
    "Telephone bell ringing": "Telephone",
    "Ambulance (siren)": "Siren",
    "Baby cry, infant cry": "Baby cry",
    "Male speech, man speaking": "Male speech",
    "Female speech, woman speaking": "Female speech",
    "Child speech, kid speaking": "Child speech",
    "Sad music": "Sad music",
    "Funny music": "Funny music",
    "Bass drum": "Bass drum",
    "Computer keyboard": "Keyboard",
    "Helicopter": "Helicopter",
    "Chicken, rooster": "Rooster",
}


def parse_cat(item: str) -> str:
    s = str(item)
    if "|" in s:
        s = s.split("|", 1)[0]
    for c in CANON_SORTED:
        if s == c or s.startswith(c):
            return c
    return "Other"


def plot_intrinsic_priority():
    q = pd.read_csv(BT / "04_q_clip.csv")
    q["cat"] = q["item"].map(parse_cat)
    q = q[q["cat"] != "Other"].copy()
    q["short"] = q["cat"].map(SHORT)
    q = q.sort_values("q").reset_index(drop=True)

    q_sub = (
        q.groupby(["cat", "short"], as_index=False)
        .agg(q_mean=("q", "mean"), q_sem=("q", "sem"), n=("q", "count"))
        .sort_values("q_mean")
    )

    deltas = pd.read_csv(BT / "01_step1_deltas.csv")
    clip = deltas[deltas["level"] == "clip"]

    fig = plt.figure(figsize=(17, 11.5))
    # Fixed positions: A and C share bottom=0.18; B/C separated by gap.
    ax0 = fig.add_axes([0.055, 0.18, 0.48, 0.74])
    ax1 = fig.add_axes([0.62, 0.62, 0.35, 0.30])
    ax2 = fig.add_axes([0.62, 0.18, 0.35, 0.30])

    # A: clip lollipop
    colors = [CAT_COLORS.get(c, "#666666") for c in q["cat"]]
    y = np.arange(len(q))
    ax0.axvline(0, color="#222222", lw=1.4, zorder=0)
    ax0.hlines(y, 0, q["q"].values, color=colors, lw=1.6, alpha=0.9)
    ax0.scatter(q["q"], y, c=colors, s=36, zorder=3, edgecolors="none")
    ax0.set_yticks([])
    ax0.set_xlabel(r"Intrinsic priority  $q_i$", fontsize=28)
    ax0.set_title("A. Clip-level priority", fontweight="bold", fontsize=30, pad=16)
    ax0.set_xlim(q["q"].min() - 0.12, q["q"].max() + 0.12)

    present = [c for c in CANON if c in set(q["cat"])]
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CAT_COLORS[c],
                    markersize=14, label=SHORT[c])
        for c in present
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.295, 0.095),
        ncol=4, frameon=False, fontsize=16, columnspacing=1.0, handletextpad=0.35,
    )

    # B: subcategory means
    y1 = np.arange(len(q_sub))
    cols1 = [CAT_COLORS[c] for c in q_sub["cat"]]
    ax1.barh(
        y1, q_sub["q_mean"], xerr=q_sub["q_sem"].fillna(0),
        color=cols1, alpha=0.92, height=0.7,
        error_kw=dict(ecolor="#222", lw=1.5, capsize=4),
    )
    ax1.axvline(0, color="#222", lw=1.3)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(q_sub["short"], fontsize=20)
    ax1.set_xlabel(r"Mean $q$ (±SEM)", fontsize=26)
    ax1.set_title("B. By subcategory", fontweight="bold", fontsize=28, pad=12)

    # C: model comparison
    schemes = ["loso", "heldout_pair"]
    scheme_lab = {"loso": "LOSO", "heldout_pair": "Held-out pair"}
    x = np.arange(len(schemes))
    w = 0.36
    m0 = [float(clip.loc[clip.scheme == s, "m0_auc"].iloc[0]) for s in schemes]
    mbt = [float(clip.loc[clip.scheme == s, "mbt_auc"].iloc[0]) for s in schemes]
    dlt = [float(clip.loc[clip.scheme == s, "delta_auc"].iloc[0]) for s in schemes]
    ax2.bar(x - w / 2, m0, w, label="Side bias only", color="#B0B0B0", edgecolor="none")
    ax2.bar(x + w / 2, mbt, w, label="Bradley–Terry", color="#4C72B0", edgecolor="none")
    for i in range(len(schemes)):
        ax2.text(i + w / 2, mbt[i] + 0.018, f"+{dlt[i]:.2f}",
                 ha="center", va="bottom", fontsize=20, fontweight="bold", color="#2F4B7C")
    ax2.set_xticks(x)
    ax2.set_xticklabels([scheme_lab[s] for s in schemes], fontsize=22)
    ax2.set_ylabel("Held-out AUC", fontsize=26)
    ax2.set_ylim(0.35, 0.88)
    ax2.set_title("C. BT vs side bias", fontweight="bold", fontsize=28, pad=12)
    ax2.legend(frameon=False, loc="upper left", fontsize=18)

    # Keep fixed canvas so A/C bottom alignment is preserved
    fig.savefig(OUT / "fig_intrinsic_priority.png", dpi=300)
    fig.savefig(OUT / "fig_intrinsic_priority.pdf")
    plt.close(fig)
    print("Wrote", OUT / "fig_intrinsic_priority.png")


def plot_eeg_beyond_acoustics():
    import json

    with open(PRIME / "04_deviation_summary.json", encoding="utf-8") as f:
        summ = json.load(f)

    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    labels = ["Acoustics\nonly", "Acoustics\n+ EEG"]
    vals = [summ["mean_auc_audio"], summ["mean_auc_audio_plus_E"]]
    bars = ax.bar(labels, vals, color=["#8C8C8C", "#C44E52"], width=0.58, edgecolor="none")
    ax.set_ylabel("LOSO AUC", fontsize=28)
    ax.set_ylim(0.55, 0.80)
    ax.set_title("A. EEG beyond acoustics", fontweight="bold", fontsize=30, pad=16)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=24, fontweight="bold")
    yb = max(vals) + 0.035
    ax.plot([0, 0, 1, 1], [vals[0] + 0.012, yb, yb, vals[1] + 0.012], color="#8B1E3F", lw=2)
    ax.text(0.5, yb + 0.008, f"ΔAUC = +{summ['delta_auc']:.3f}",
            ha="center", va="bottom", fontsize=24, fontweight="bold", color="#8B1E3F")
    fig.tight_layout()
    fig.savefig(OUT / "fig_eeg_beyond_acoustics.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_eeg_beyond_acoustics.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT / "fig_eeg_beyond_acoustics.png")


if __name__ == "__main__":
    plot_intrinsic_priority()
    plot_eeg_beyond_acoustics()
    print("All figures in", OUT)
