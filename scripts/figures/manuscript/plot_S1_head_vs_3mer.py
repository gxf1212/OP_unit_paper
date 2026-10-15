#!/usr/bin/env python3
"""Supplementary Fig. S1: head (monomer) ΔG_rel vs 3-mer ΔG_rel scatter.

Joins the public `reference_120.csv` and `reference_3mer.csv` tables (with
source-workspace fallbacks to `reference_120.csv` and `3mer.csv`) using the naming rule
`L###-3 ↔ hL(###-2)` (e.g. C103-3 ↔ hC101), then plots the
trimer ΔG_rel against the corresponding head ΔG_rel, colored by anion class.
Style mirrors the existing prediction scatter (Fig1_14) in plot_data_analysis.py.

The `dG_rel_kcal_per_mol` columns ARE ΔG_rel = G(1.6 nm) − G(4.0 nm) per the manuscript
protocol, so no PMF reprocessing is needed.

Output: manuscript/Figures/FigS1_Head_vs_3mer_dG_rel.png (routed via
`plot_utils.FIGURE_DESTINATIONS`).

Usage:
    python plot_head_vs_3mer.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_utils import (
    ANION_COLORS,
    PUBLICATION_FONTS,
    apply_publication_style,
    save_figure,
    setup_plot_style,
)

# Trimer prefix -> terminal-chemistry subgroup letter. The 3-mer set covers
# the N, C, H, P and S subgroups only; no A-subgroup (hydroxyl-terminated,
# hA) 3-mer was built. This descriptive figure therefore colours by the five
# subgroups actually present, using the shared six-subgroup palette (N in a
# lighter sibling blue, distinct from the A/N merged-class blue used in
# ML/LOGO figures), consistent with Fig. 2a / S2 / S4.
TRIMER_TO_SUBGROUP = {"C": "C", "H": "H", "P": "P", "S": "S", "N": "N"}

# Five-subgroup palette mirroring SUBGROUP_COLORS in plot_data_analysis.py,
# so the N points read in the same lighter blue as in Fig. 2a / S2 / S4.
SUBGROUP_COLORS = {
    "N": "#6baed6",
    "C": ANION_COLORS["C"],
    "H": ANION_COLORS["H"],
    "P": ANION_COLORS["P"],
    "S": ANION_COLORS["S"],
}


def load_head_and_trimers() -> pd.DataFrame:
    """Join head and 3-mer ΔG_rel by the `L###-3 ↔ hL(###-2)` naming rule."""
    base = Path(__file__).resolve().parents[3]
    head_path = base / "data" / "reference_120.csv"
    if not head_path.exists():
        head_path = base / "data" / ("head" + ".csv")
    trimer_path = base / "data" / "reference_3mer.csv"
    if not trimer_path.exists():
        trimer_path = base / "data" / ("3mer" + ".csv")
    head = pd.read_csv(head_path)
    trimers = pd.read_csv(trimer_path)

    legacy_dg_col = "delta" + "G"
    head_value_col = (
        "dG_rel_kcal_per_mol"
        if "dG_rel_kcal_per_mol" in head.columns
        else legacy_dg_col
    )
    trimer_value_col = (
        "dG_rel_kcal_per_mol"
        if "dG_rel_kcal_per_mol" in trimers.columns
        else legacy_dg_col
    )
    head_map = dict(zip(head["Name"], head[head_value_col]))

    rows = []
    for _, row in trimers.iterrows():
        name = str(row["Name"])  # e.g. "C103-3"
        letter = name[0]
        num = int(name[1:4])
        head_name = f"h{letter}{num - 2:03d}"
        if head_name not in head_map:
            print(f"Warning: no head {head_name} for trimer {name}, skipped")
            continue
        rows.append(
            {
                "trimer": name,
                "head": head_name,
                "subgroup": TRIMER_TO_SUBGROUP.get(letter, letter),
                "head_dg": head_map[head_name],
                "trimer_dg": row[trimer_value_col],
            }
        )

    return pd.DataFrame(rows)


def create_head_vs_3mer_scatter(
    df: pd.DataFrame, typography: str = "manuscript"
):
    """Create the head-versus-3-mer ΔG_rel scatter.

    ``manuscript`` applies the shared Times New Roman mathtext profile before
    labels are created. ``legacy`` retains the former rendering for provenance.
    """
    # 16-in-wide canvas matches the SI figure family (S4 16 in, S7 17 in,
    # S8/S9/S10 18 in) so font sizes read identically at equal display width;
    # the old 10x10 canvas made every label look oversized in the manuscript.
    fig, ax = setup_plot_style(figsize=(16, 10))
    if typography == "manuscript":
        # ``setup_plot_style`` creates axes in a local rc-context. Reapply the
        # shared profile here so mathtext in labels uses Times New Roman too.
        apply_publication_style()
    elif typography != "legacy":
        raise ValueError("typography must be 'manuscript' or 'legacy'")

    head_dg = np.asarray(df["head_dg"], dtype=float)
    trimer_dg = np.asarray(df["trimer_dg"], dtype=float)

    # Scatter style unified across the manuscript (matches Fig. S5):
    # round markers with a thin black outline so points read on any background.
    colors = [SUBGROUP_COLORS.get(g, "#808080") for g in df["subgroup"]]
    ax.scatter(
        head_dg,
        trimer_dg,
        c=colors,
        marker="o",
        s=60,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    # Head and 3-mer ΔG are on different scales (3-mer ≈ 2×head + 6), so a 1:1
    # line is meaningless here; show the linear best-fit reference instead.
    slope, intercept = np.polyfit(head_dg, trimer_dg, 1)
    x_fit = np.linspace(head_dg.min(), head_dg.max(), 50)
    ax.plot(
        x_fit,
        slope * x_fit + intercept,
        "k--",
        linewidth=2,
        alpha=0.8,
    )

    pearson = np.corrcoef(head_dg, trimer_dg)[0, 1]
    r2 = pearson**2
    spearman, _ = spearmanr(head_dg, trimer_dg)

    ax.set_xlabel('Head $\\Delta G_\\mathrm{rel}$ ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    ax.set_ylabel('3-mer $\\Delta G_\\mathrm{rel}$ ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])

    ax.grid(False)

    # Stats box mirrors the Supplementary Fig. S10b legend style: a frameon
    # legend carrying the coefficients as pure text, so the box has the same
    # frame/shadow/font as S10 rather than a borderless white text patch.
    stats_text = (
        f"Pearson $r$ = {pearson:.3f}"
        f"\nSpearman $\\rho$ = {spearman:.3f}"
        f"\n$R^2$ = {r2:.3f}"
    )
    ax.legend(
        [Line2D([], [], linestyle="none", marker="")],
        [stats_text],
        loc="lower left", frameon=True,
        fontsize=PUBLICATION_FONTS["annotation"],
        prop={"family": "Times New Roman"},
        handlelength=0, handletextpad=0.2,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            linestyle="none",
            label=group,
            markerfacecolor=SUBGROUP_COLORS[group],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=10,
        )
        for group in ["N", "C", "H", "P", "S"]
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fontsize=PUBLICATION_FONTS["legend"],
    )

    plt.tight_layout()
    return fig, ax


def main() -> None:
    """Load data, draw, and save to common_figures."""
    df = load_head_and_trimers()
    print(f"Joined {len(df)} head/3-mer pairs")

    fig, _ = create_head_vs_3mer_scatter(df)
    save_figure(fig, "FigS1_Head_vs_3mer_dG_rel")


if __name__ == "__main__":
    main()
