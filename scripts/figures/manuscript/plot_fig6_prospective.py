#!/usr/bin/env python3
"""Generate Figure 6: Prospective evaluation of newly designed amine-oxide motifs.

Three panels:
  A. 2 x 4 chemical-structure grid of the eight prospectively evaluated
     amine-oxide motifs, ordered by increasing ensemble-predicted dG_rel
     (predicted rank 1-8).
  B. Rank-flow (slopegraph): ensemble-predicted rank (left) to PMF-derived rank
     (right) for each candidate; Spearman rho = 0.833.
  C. Ensemble-predicted versus PMF-derived dG_rel scatter with the identity line
     and two error bars: horizontal = ensemble SD (inter-model disagreement),
     vertical = PMF replicate SD (across five independent repeats).

Data:
  - processed table: data/prospective_pmf_8.csv

Output: manuscript/Figures/Fig6_Prospective_Evaluation.png
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[3]  # up to OP_unit/
sys.path.insert(0, str(BASE / "scripts" / "figures"))

from plot_utils import (  # noqa: E402
    PUBLICATION_FONTS,
    apply_publication_style,
    save_publication_figure,
)
from public_data import load_prospective_pmf

CANDIDATES = ["N035", "N055", "N067", "N103", "N123", "N127", "N143", "N159"]

# Categorical colours for the eight candidates (distinct, print-friendly).
CANDIDATE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# Figure-local font sizes. The earlier 0.78 scale was tuned for the short
# 22x15 layout; at the current 22x18 size the B/C axes are as large as a
# normal single-panel figure, so the shared PUBLICATION_FONTS defaults apply
# unmodified (matching the other manuscript figures).
FONT_SCALE = 1.0
LOCAL_FONTS = {
    key: max(10, round(value * FONT_SCALE))
    for key, value in PUBLICATION_FONTS.items()
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prospective_data() -> pd.DataFrame:
    """Merge ensemble predictions and PMF results for the eight candidates."""
    df = load_prospective_pmf().copy()
    df = df[df["Name"].isin(CANDIDATES)].copy()
    df = df.rename(columns={
        "predicted_dG_rel": "ensemble_mean",
        "PMF_dG_rel": "pmf_dG",
        "PMF_replicate_SD": "pmf_sd",
    })
    df = df.sort_values("ensemble_mean").reset_index(drop=True)
    df["pred_rank"] = np.arange(1, len(df) + 1)
    df["pmf_rank"] = df["pmf_dG"].rank(method="min").astype(int)
    df["signed_err"] = df["ensemble_mean"] - df["pmf_dG"]
    return df


# ---------------------------------------------------------------------------
# Panel A: chemical structures
# ---------------------------------------------------------------------------
def render_smiles(
    smiles: str,
    width: int = 600,
    height: int = 460,
    uniform_scale: bool = True,
    target_span: float = 24.0,
) -> np.ndarray:
    """Render a SMILES to an RGB array via RDKit's Cairo drawer.

    With ``uniform_scale=True`` the 2D conformer is translated to the origin
    and rescaled so every molecule spans the same coordinate range
    (``target_span``). Because the drawer fits each molecule to the fixed
    canvas, this makes bond lengths visually equal across molecules of
    different sizes and centres every structure on its canvas. Set
    ``uniform_scale=False`` to reproduce the original per-molecule fit.
    """
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    if uniform_scale:
        conf = mol.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span > 0.0:
            scale = target_span / span
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(
                    i, Point3D((p.x - cx) * scale, (p.y - cy) * scale, 0.0)
                )

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 4
    opts.padding = 0.12
    # Atom labels use fixedFontSize (exact pixels) because baseFontSize is
    # non-monotonic in this rdkit build (capped/erratic past ~40 px) and
    # fontSize/fontFileScale are NOT valid MolDrawOptions (silently ignored).
    opts.fixedFontSize = 50
    opts.additionalAtomLabelPadding = 0.2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return plt.imread(io.BytesIO(png))  # (H, W, 3) RGB


def panel_a_structures(fig, ax_a, df: pd.DataFrame) -> None:
    """Draw the 2 x 4 structure grid inside the top panel axes."""
    ax_a.set_axis_off()
    pos = ax_a.get_position()  # figure-fraction bbox of the top panel

    n_rows, n_cols = 2, 4
    cell_w = pos.width / n_cols
    cell_h = pos.height / n_rows

    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, n_cols)
        x0 = pos.x0 + c * cell_w
        y0 = pos.y0 + (n_rows - 1 - r) * cell_h  # top row first

        # molecule image in the upper part of the cell
        img = render_smiles(str(row["SMILES"]))
        ax_img = fig.add_axes([x0 + cell_w * 0.03, y0 + cell_h * 0.22,
                               cell_w * 0.94, cell_h * 0.76])
        ax_img.imshow(img)
        ax_img.set_axis_off()

        # candidate ID beneath the structure (the predicted rank is conveyed
        # by panel B's slopegraph, so it is not repeated here). Fixed 2x the
        # default body text so the labels stay readable at figure scale.
        fig.text(
            x0 + cell_w / 2,
            y0 + cell_h * 0.03,
            str(row["Name"]),
            ha="center",
            va="bottom",
            fontsize=32,
            fontfamily="Times New Roman",
        )

    ax_a.text(
        -0.08, 1.15, "a",
        transform=ax_a.transAxes, fontsize=40,
        fontweight="bold", va="top", ha="left", fontfamily="Times New Roman",
    )


# ---------------------------------------------------------------------------
# Panel B: rank-flow (slopegraph)
# ---------------------------------------------------------------------------
def panel_b_rank_flow(ax, df: pd.DataFrame) -> None:
    """Slopegraph connecting predicted rank (left) to PMF rank (right)."""
    pred_pos = 0.0
    pmf_pos = 1.0

    for i, (_, row) in enumerate(df.iterrows()):
        y0, y1 = row["pred_rank"], row["pmf_rank"]
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        ax.plot(
            [pred_pos, pmf_pos], [y0, y1], "-o",
            color=color, linewidth=2.5, markersize=9,
            markeredgecolor="black", markerfacecolor=color, zorder=3,
        )
        ax.text(
            pred_pos - 0.06, y0, row["Name"],
            ha="right", va="center",
            fontsize=LOCAL_FONTS["value_text"], fontfamily="Times New Roman",
        )
        ax.text(
            pmf_pos + 0.06, y1, row["Name"],
            ha="left", va="center",
            fontsize=LOCAL_FONTS["value_text"], fontfamily="Times New Roman",
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.6, 8.4)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(1, 9))
    ax.tick_params(axis="y", labelsize=LOCAL_FONTS["tick_label"])
    ax.set_xticks([pred_pos, pmf_pos])
    ax.set_xticklabels(["Predicted rank", "PMF rank"],
                       fontsize=LOCAL_FONTS["tick_label"],
                       fontfamily="Times New Roman")
    ax.set_ylabel(r"$\Delta G_{\mathrm{rel}}$ rank",
                  fontfamily="Times New Roman",
                  fontsize=LOCAL_FONTS["axis_label"])

    # Spearman rho is reported in the Results text and the figure caption, not
    # inside the panel (the in-figure box overlapped the slope lines).

    ax.text(
        -0.16, 1.15, "b",
        transform=ax.transAxes, fontsize=40,
        fontweight="bold", va="top", ha="left", fontfamily="Times New Roman",
    )


# ---------------------------------------------------------------------------
# Panel C: predicted vs PMF scatter
# ---------------------------------------------------------------------------
def panel_c_scatter(ax, df: pd.DataFrame) -> None:
    """Predicted vs PMF-derived dG_rel with identity line and error bars."""
    x = df["ensemble_mean"].to_numpy()
    y = df["pmf_dG"].to_numpy()
    xerr = df["ensemble_sd"].to_numpy()
    yerr = df["pmf_sd"].to_numpy()

    # Axis limits span the error-bar extents (not just the point positions) so
    # the large N159 error bar is fully visible rather than clipped at the axes.
    # The identity line follows the same range.
    lo = min((x - xerr).min(), (y - yerr).min()) - 0.4
    hi = max((x + xerr).max(), (y + yerr).max()) + 0.4
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, zorder=1)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row["ensemble_mean"], row["pmf_dG"],
            xerr=row["ensemble_sd"], yerr=row["pmf_sd"],
            fmt="o", ms=11, color=CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)],
            ecolor="black", elinewidth=1.5, capsize=4,
            markeredgecolor="black", zorder=3,
        )
        # No per-point name labels: candidate colours match panel B, which
        # identifies each candidate by name on both sides of the slopegraph.
        # The figure caption states the colour correspondence.

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=LOCAL_FONTS["tick_label"])
    ax.set_xlabel(
        r"Predicted $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        fontfamily="Times New Roman", fontsize=LOCAL_FONTS["axis_label"],
    )
    ax.set_ylabel(
        r"PMF-derived $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        fontfamily="Times New Roman", fontsize=LOCAL_FONTS["axis_label"],
    )

    # Stats box mirrors the Supplementary Fig. S10b legend style: a frameon
    # legend carrying the coefficients as pure text, so the box has the same
    # frame/shadow/font as S10/S1/S7/S8b rather than a borderless white patch.
    # nudged right from the old 0.02 so it clears the y-axis label area.
    ax.legend(
        [Line2D([], [], linestyle="none", marker="")],
        ["Pearson $r$ = 0.791\n"
         r"MAE = 1.54 $\mathrm{kcal\cdot mol^{-1}}$"],
        loc="upper left", bbox_to_anchor=(0.05, 0.90),
        frameon=True,
        fontsize=LOCAL_FONTS["annotation"],
        prop={"family": "Times New Roman"},
        handlelength=0, handletextpad=0.2,
    )

    ax.text(
        -0.28, 1.15, "c",
        transform=ax.transAxes, fontsize=40,
        fontweight="bold", va="top", ha="left", fontfamily="Times New Roman",
    )


def main() -> None:
    apply_publication_style()
    df = load_prospective_data()
    print(df[["Name", "ensemble_mean", "pmf_dG", "pred_rank", "pmf_rank", "signed_err"]].to_string(index=False))

    # Tall layout: the bottom row (rank flow + scatter) gets the larger
    # height ratio so panel C approaches a square axes and the eight
    # candidates spread out vertically instead of crowding.
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0],
                          hspace=0.30, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a_structures(fig, ax_a, df)
    panel_b_rank_flow(ax_b, df)
    panel_c_scatter(ax_c, df)

    out_path = BASE / "manuscript" / "Figures" / "Fig6_Prospective_Evaluation.png"
    save_publication_figure(fig, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
