"""Manuscript Supplementary Fig. S3 (2x2): additional physicochemical
descriptors versus PMF-derived relative membrane affinity.

Four descriptor panels coloured by the six terminal subgroups (A, N, C, H, P,
S) with the manuscript SUBGROUP_COLORS: logP, Molecular Weight, Max Partial
Charge (a charge-localization descriptor, not a dipole proxy) and Labute ASA.
The original Figure 2d logP scatter is folded into this panel set, making S4 a
2x2 grid that keeps the same bulk-descriptor story (logP / MW / charge / ASA all
show subgroup-dependent behaviour and cannot each alone explain the dataset).

Each panel reuses the exact Figure 2 tile geometry (axes box 6.30 x 4.725 in,
full four-spine frame, Times New Roman, axis labels at PUBLICATION_FONTS) and
the four tiles are stitched by ``_tile_assembler.assemble_2x2`` using the same
PAD / TOP / GAP / panel-letter geometry as assemble_fig2.py, so the figure is
pixel-identical in style to Figure 2 and the other supplementary figures.

Output: manuscript/Figures/FigS3_Descriptor_Scatters_6subgroups.png
(overwrites the former 3-panel S4 figure)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_data_analysis import (
    SUBGROUP_COLORS,
    SUBGROUP_ORDER,
    compute_structure_descriptors,
    load_molecular_data,
)
from _tile_assembler import make_tile, style_tile, assemble_2x2
from plot_utils import PUBLICATION_FONTS

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_NAME = "FigS3_Descriptor_Scatters_6subgroups"

# (panel letter, descriptor column, x label) reading order.
PANELS = [
    ("a", "logP", "logP"),
    ("b", "mol_wt", "Molecular Weight (Da)"),
    ("c", "max_charge", "Max Partial Charge (|e|)"),
    ("d", "ASA", "Labute ASA (\u00C5\u00B2)"),
]


def _extract_subgroup(name: str) -> str:
    return str(name)[1].upper()


def _draw_tile(col: str, xlabel: str, df: pd.DataFrame):
    """Build one letter-free Figure 2 tile for a single descriptor."""
    fig, ax = make_tile()
    for code in SUBGROUP_ORDER:
        part = df[df["subgroup"] == code]
        ax.scatter(part[col], part["dG_rel_kcal_per_mol"], c=SUBGROUP_COLORS[code], s=55,
                   alpha=0.8, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xlabel(xlabel, fontsize=PUBLICATION_FONTS["axis_label"],
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
    df = load_molecular_data().copy()
    desc = df["SMILES"].apply(compute_structure_descriptors)
    df[["mol_wt", "logP", "ASA", "max_charge", "n_rotatable"]] = pd.DataFrame(
        desc.tolist(), index=df.index)
    df["subgroup"] = df["Name"].apply(_extract_subgroup)

    tile_figs = [_draw_tile(col, xlabel, df) for _, col, xlabel in PANELS]
    letters = [letter for letter, _, _ in PANELS]

    legend = [
        ("patch", SUBGROUP_COLORS[code], code) for code in SUBGROUP_ORDER
    ]

    out_path = BASE_DIR / "manuscript" / "Figures" / f"{OUT_NAME}.png"
    assemble_2x2(tile_figs, letters, legend, out_path,
                 legend_position='bottom')


if __name__ == "__main__":
    main()
