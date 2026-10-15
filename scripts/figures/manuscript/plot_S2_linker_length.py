"""Supplementary Figure S2: Linker-length effect on membrane affinity (2x2 grid).

For each terminal-chemistry class (C, H, P, S) a panel shows the matched design:
every thin line connects the same R-group across its linker-length m values,
isolating the effect of changing m while holding terminal chemistry and R-group
identity constant. A black line with square markers indicates the mean across the
eight R-groups at each m. Panels a-d share one y-axis so the terminal-chemistry energy
scale stays comparable.

Output: manuscript/Figures/FigS2_Linker_Length_Effect.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_data_analysis import (
    R_GROUP_MAP,
    R_GROUP_ORDER,
    R_GROUP_COLORS,
    R_GROUP_SHORT_LABELS,
    load_molecular_data,
)
from _tile_assembler import make_tile, style_tile, assemble_2x2
from plot_utils import PUBLICATION_FONTS

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_NAME = "FigS2_Linker_Length_Effect"

# (panel letter, terminal-chemistry code) per 2x2 cell, reading order.
# A_N is the mixed O / O- class and is excluded from every panel.
PANELS = [("a", "C"), ("b", "H"), ("c", "P"), ("d", "S")]

PREFIX_TO_GROUP = {
    "hC": "C", "hH": "H", "hP": "P", "hS": "S",
    "hA": "A_N", "hN": "A_N",
}


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


def _draw_tile(group: str, df: pd.DataFrame, y_lo, y_hi):
    """Build one letter-free Figure 2 tile for a terminal-chemistry class."""
    fig, ax = make_tile()
    df = df[df["group"] == group]

    for rg in R_GROUP_ORDER:
        rg_sub = df[df["rgroup"] == rg].sort_values("m")
        if rg_sub.empty:
            continue
        ax.plot(rg_sub["m"], rg_sub["dG_rel_kcal_per_mol"], marker="o", markersize=5,
                linewidth=1.6, color=R_GROUP_COLORS[rg], alpha=0.85, zorder=3)

    means = df.groupby("m")["dG_rel_kcal_per_mol"].mean()
    ms = means.index.values
    ax.plot(ms, means.values, color="black", linewidth=2.0, marker="s",
            markersize=6, zorder=5, label="Mean")

    ax.set_xticks(sorted(df["m"].unique()))
    ax.set_xlim(min(df["m"].min(), 1) - 0.4, df["m"].max() + 0.4)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Linker length $m$", fontsize=PUBLICATION_FONTS["axis_label"],
                  fontweight="bold", fontfamily="Times New Roman", labelpad=14)
    ax.set_ylabel(
        r"$\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        fontsize=PUBLICATION_FONTS["axis_label"], fontweight="bold",
        fontfamily="Times New Roman", labelpad=12)
    style_tile(ax)
    ax.grid(False)
    ax.set_axisbelow(True)
    return fig


def main() -> None:
    df = load_molecular_data()
    decoded = df["Name"].apply(decode_motif)
    df["group"] = [d[0] for d in decoded]
    df["rgroup"] = [d[1] for d in decoded]
    df["m"] = [d[2] for d in decoded]
    df = df.dropna(subset=["group", "rgroup", "m"])
    df = df[df["group"].isin([g for _, g in PANELS])]

    y_min = df["dG_rel_kcal_per_mol"].min()
    y_max = df["dG_rel_kcal_per_mol"].max()
    y_pad = 0.12 * (y_max - y_min)
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    tile_figs = [_draw_tile(group, df, y_lo, y_hi) for _, group in PANELS]
    letters = [letter for letter, _ in PANELS]

    legend = [
        ("line", R_GROUP_COLORS[rg], R_GROUP_SHORT_LABELS[rg])
        for rg in R_GROUP_ORDER
    ]
    legend.append(("line", "black", "Mean"))

    out_path = BASE_DIR / "manuscript" / "Figures" / f"{OUT_NAME}.png"
    assemble_2x2(tile_figs, letters, legend, out_path,
                 legend_position='bottom')


if __name__ == "__main__":
    main()
