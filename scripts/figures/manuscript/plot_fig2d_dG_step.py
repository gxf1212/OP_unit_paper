"""Manuscript Figure 2d tile: per-step change in relative membrane affinity
with linker length m, summarised as boxplots per terminal-chemistry class.

This panel answers the matched-design question left open by the linker-length
line plots: holding terminal chemistry and R-group identity constant, by how
much does adding one methylene to the linker shift DeltaG_rel, and does the
direction depend on the terminal chemistry? For each class we collect the
within-R-group step differences DeltaG_rel(m+1) - DeltaG_rel(m) and show their
distribution as a boxplot. A dashed zero line marks no net change.

The tile reuses the exact Figure 2 geometry (7.8 x 7.475 in canvas, axes box
6.30 x 4.725 in, full four-spine frame, Times New Roman) so assemble_fig2.py
can paste it 1:1 next to Fig2a/2b/2c. Panel letter is added by the assembler,
so this tile is letter-free, matching Fig2c/Fig2d tiles.

Output: manuscript/Figures/Fig2d_dG_step_boxplot.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_data_analysis import (
    R_GROUP_MAP,
    R_GROUP_ORDER,
    ANION_COLORS,
    load_molecular_data,
)
from _fig2_tile_base import make_tile as _new_tile, style_tile as _style_axes
from plot_utils import PUBLICATION_FONTS

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_NAME = "Fig2d_dG_step_boxplot"

PREFIX_TO_GROUP = {
    "hC": "C", "hH": "H", "hP": "P", "hS": "S",
    "hA": "A_N", "hN": "A_N",
}
# Boxplot classes: the four terminal chemistries with >=3 m values (the
# matched-line panels). A_N carries only two linker lengths and is omitted.
BOX_GROUPS = ["C", "H", "P", "S"]


def decode_motif(name: str):
    prefix = name[:2]
    group = PREFIX_TO_GROUP.get(prefix)
    if group is None:
        return None, None, None
    idx = int(name[2:])
    k = (idx - 1) // 4
    rgroup = R_GROUP_MAP[((idx - 1) % 32) + 1]
    m = (k // 8) + 1 if group == "C" else (k // 8) + 2
    return group, rgroup, m


def main() -> None:
    df = load_molecular_data()
    decoded = df["Name"].apply(decode_motif)
    df["group"] = [d[0] for d in decoded]
    df["rgroup"] = [d[1] for d in decoded]
    df["m"] = [d[2] for d in decoded]
    df = df.dropna(subset=["group", "rgroup", "m"])

    step_by_group = {g: [] for g in BOX_GROUPS}
    for group in BOX_GROUPS:
        sub = df[df["group"] == group]
        for rg in R_GROUP_ORDER:
            rg_sub = sub[sub["rgroup"] == rg].sort_values("m")
            if len(rg_sub) < 2:
                continue
            step_by_group[group].extend(np.diff(rg_sub["dG_rel_kcal_per_mol"].values).tolist())

    fig, ax = _new_tile()
    bp = ax.boxplot(
        [step_by_group[g] for g in BOX_GROUPS],
        positions=range(1, len(BOX_GROUPS) + 1),
        widths=0.55, patch_artist=True, showmeans=False,
    )
    for patch, g in zip(bp["boxes"], BOX_GROUPS):
        patch.set_facecolor(ANION_COLORS[g])
        patch.set_edgecolor("black")
        patch.set_alpha(0.6)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.2)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.8)

    # Overlay individual data points with jitter (same style as 2a/2c)
    rng = np.random.default_rng(42)
    for idx, g in enumerate(BOX_GROUPS, 1):
        steps = step_by_group[g]
        jitter = rng.normal(0, 0.04, size=len(steps))
        ax.scatter(np.full_like(steps, idx) + jitter, steps,
                   color=ANION_COLORS[g], edgecolors='black',
                   alpha=0.5, s=35, linewidths=0.4, zorder=3)

    ax.axhline(0, color="grey", linestyle="--", linewidth=1.6, zorder=0)

    ax.set_xticks(range(1, len(BOX_GROUPS) + 1))
    # Single-letter codes only (no chemical formulas inside the figure), decoded
    # in the caption per the manuscript notation decision. The tick size is set
    # by _style_axes (PUBLICATION_FONTS tick_label); leave it to that.
    ax.set_xticklabels(BOX_GROUPS)
    ax.set_xlabel("Terminal chemistry",
                  fontsize=PUBLICATION_FONTS["axis_label"], fontweight="bold",
                  fontfamily="Times New Roman", labelpad=14)
    # The two-term y label is much longer than the other tiles'; it is set one
    # step smaller than PUBLICATION_FONTS axis_label so it fits the fixed
    # canvas instead of being clipped at both ends.
    ax.set_ylabel(
        r"$\Delta G_{\mathrm{rel,(m+1)}} - \Delta G_{\mathrm{rel,(m)}}$ "
        r"($\mathrm{kcal\cdot mol^{-1}}$)",
        fontsize=PUBLICATION_FONTS["axis_label"] - 8, fontweight="bold",
        fontfamily="Times New Roman", labelpad=12)
    _style_axes(ax)
    ax.grid(False)
    ax.set_axisbelow(True)
    # Tight y-range around the step values so the zero line is meaningful.
    allv = np.concatenate([step_by_group[g] for g in BOX_GROUPS])
    lo, hi = allv.min(), allv.max()
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    fig.savefig(str(BASE_DIR / "manuscript" / "Figures" / OUTPUT_NAME) + ".png",
                dpi=300, bbox_inches=None, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {OUTPUT_NAME}.png")
    print("Per-group medians:",
          {g: float(np.median(step_by_group[g])) for g in BOX_GROUPS})


if __name__ == "__main__":
    main()
