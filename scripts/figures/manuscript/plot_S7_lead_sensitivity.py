#!/usr/bin/env python3
"""Supplementary Fig. S7: sensitivity of few-shot LOGO to lead-motif selection.

Builds the manuscript SI figure from the exhaustive lead-sensitivity database
(``data/lead_sensitivity.csv``, falling back to the SQLite export), which
contains all 696 possible lead sets (16 single, 120 pairs, 560 triples) drawn
from the 16 A/N motifs. Every combination was evaluated with the
representation-model combination selected under strict LOGO for the A/N fold
(datamol_secfp + XGBoost), i.e. the same protocol as the few-shot LOGO analysis
except that the leads are varied.

Panels (2 x 2; distributions share the left-column width, scatters share the
right-column width):
    (a) External test Pearson r distribution versus the number of lead motifs,
        with the best achievable combination marked explicitly.
    (b) Predicted versus PMF-derived dG_rel for the best three-lead
        combination by Pearson r (hA013, hN001, hN029; r = 0.85,
        MAE = 1.24 kcal/mol).
    (c) External test MAE distribution versus the number of lead motifs; the
        diamond marks the best (lowest-MAE) combination and the star marks the
        MAE of the best-by-Pearson-r combination, showing that the two
        criteria dissociate.
    (d) Same as (b) for the best two-lead combination by Pearson r
        (hA017, hN029; r = 0.76, MAE = 0.59 kcal/mol).

Notation follows the manuscript decision of 2026-09-02: ML/LOGO figures use the
five merged classes; this figure concerns the A/N class only, and no chemical
formulas appear inside the figure (the caption decodes the identifiers).

Output: manuscript/Figures/FigS7_Lead_Sensitivity.png (routed via
``plot_utils.FIGURE_DESTINATIONS``).

Usage:
    python plot_S7_lead_sensitivity.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_utils import (
    PUBLICATION_FONTS,
    format_axis_labels,
    get_publication_style,
    save_figure,
)
from public_data import load_lead_sensitivity

VIOLIN_COLOR = "#1f77b4"  # matches the Pearson r colour used across lead-sweep figures
STAR_COLOR = "#DD8452"    # best-by-r combination marker in the MAE panel

# Strict-LOGO A/N fold performance (Table 1), drawn as a reference baseline in
# the distribution panels: external Pearson r = 0.536 and external MAE = 2.13
# kcal/mol for the held-out A/N class with no few-shot augmentation.
STRICT_LOGO_R = 0.536
STRICT_LOGO_MAE = 2.13


def load_results() -> pd.DataFrame:
    """Load all exhaustive lead-sensitivity combinations."""
    df = load_lead_sensitivity()
    if df.empty:
        raise ValueError("No rows found in the lead-sensitivity database")
    return df


def plot_lead_count_distribution(ax, df: pd.DataFrame, value_col: str,
                                 ylabel: str, best: str,
                                 baseline: float = None,
                                 extra_marker: bool = False,
                                 star_best_mae: bool = False,
                                 legend_loc: str = "lower left",
                                 legend_frameon: bool = False) -> None:
    """Distribution panels (a) and (b): metric versus number of lead motifs.

    ``best`` selects which extreme the diamond marks ('max' for Pearson r,
    'min' for MAE). With ``extra_marker`` (panel b) a star additionally marks
    the metric value of the best-by-Pearson-r combination at each lead count,
    so the r-versus-MAE dissociation is visible in one panel.
    """
    lead_counts = sorted(df["lead_count"].unique())
    data_by_count = [
        df.loc[df["lead_count"] == lc, value_col].dropna().values
        for lc in lead_counts
    ]

    parts = ax.violinplot(
        data_by_count,
        positions=lead_counts,
        widths=0.6,
        showmeans=True,
        showextrema=False,
        showmedians=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(VIOLIN_COLOR)
        pc.set_alpha(0.3)

    # Jittered individual combinations on top of each violin.
    rng = np.random.default_rng(42)
    for lc, data in zip(lead_counts, data_by_count):
        ax.scatter(
            rng.normal(lc, 0.07, size=len(data)),
            data,
            alpha=0.35,
            s=28,
            color=VIOLIN_COLOR,
            edgecolor="none",
        )

    pick = (lambda d: float(np.max(d))) if best == "max" else (lambda d: float(np.min(d)))
    for lc, data in zip(lead_counts, data_by_count):
        mean_val = float(np.mean(data))
        best_val = pick(data)
        ax.annotate(
            f"{mean_val:.2f}",
            (lc, mean_val),
            textcoords="offset points",
            xytext=(22, -4),
            fontsize=PUBLICATION_FONTS["annotation"],
            fontfamily="Times New Roman",
        )
        ax.scatter(
            [lc],
            [best_val],
            marker="D",
            s=110,
            color="black",
            zorder=5,
        )
        ax.annotate(
            f"{best_val:.2f}",
            (lc, best_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=PUBLICATION_FONTS["annotation"],
            fontfamily="Times New Roman",
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=VIOLIN_COLOR,
               alpha=0.6, markersize=10, label="Individual lead set"),
        Line2D([0], [0], marker="D", linestyle="none", color="black",
               markersize=10, label="Best combination"),
    ]
    if extra_marker:
        # Star: the metric value of the combination that maximises Pearson r
        # within each lead-count group (panel b only).
        best_r_values = [
            float(df.loc[df.loc[df["lead_count"] == lc, "pearson_r"].idxmax(),
                         value_col])
            for lc in lead_counts
        ]
        ax.scatter(lead_counts, best_r_values, marker="*", s=260,
                   color=STAR_COLOR, edgecolor="black", linewidth=0.6,
                   zorder=6)
        for lc, val in zip(lead_counts, best_r_values):
            ax.annotate(
                f"{val:.2f}",
                (lc, val),
                textcoords="offset points",
                xytext=(-12, 0),
                ha="right",
                fontsize=PUBLICATION_FONTS["annotation"],
                fontfamily="Times New Roman",
            )
        handles.append(
            Line2D([0], [0], marker="*", linestyle="none", color=STAR_COLOR,
                   markeredgecolor="black", markersize=14,
                   label="Best-$r$ combination")
        )

    if star_best_mae:
        # Star: the Pearson r attained by the lowest-MAE combination within each
        # lead-count group (panel a only), mirroring the best-r MAE star in (c).
        best_mae_r = [
            float(df.loc[df.loc[df["lead_count"] == lc, "mae"].idxmin(),
                         "pearson_r"])
            for lc in lead_counts
        ]
        ax.scatter(lead_counts, best_mae_r, marker="*", s=260,
                   color=STAR_COLOR, edgecolor="black", linewidth=0.6,
                   zorder=6)
        for lc, val in zip(lead_counts, best_mae_r):
            ax.annotate(
                f"{val:.2f}",
                (lc, val),
                textcoords="offset points",
                xytext=(-12, 0),
                ha="right",
                fontsize=PUBLICATION_FONTS["annotation"],
                fontfamily="Times New Roman",
            )
        handles.append(
            Line2D([0], [0], marker="*", linestyle="none", color=STAR_COLOR,
                   markeredgecolor="black", markersize=14,
                   label="Best-MAE combination")
        )

    if baseline is not None:
        ax.axhline(baseline, color="0.5", linestyle="--", linewidth=1.3,
                   zorder=1)
    ax.set_xticks(lead_counts)
    ax.set_xticklabels([str(lc) for lc in lead_counts])
    ax.set_xlim(0.4, 3.8)
    ax.grid(False)

    format_axis_labels(
        ax,
        xlabel="Number of lead motifs",
        ylabel=ylabel,
        title=None,
    )
    ax.legend(
        handles=handles,
        loc=legend_loc,
        frameon=legend_frameon,
        fancybox=False,
        framealpha=0.85,
        edgecolor="0.8",
        fontsize=18,
        prop={"family": "Times New Roman", "size": 18},
    )


def plot_best_combination_scatter(ax, row: pd.Series) -> None:
    """Scatter panels (b, d): predicted vs measured dG_rel for one lead set."""
    predictions = np.asarray(json.loads(str(row["predictions"])), dtype=float)
    true_values = np.asarray(json.loads(str(row["true_values"])), dtype=float)

    # Scatter style unified across the manuscript (matches Fig. S5).
    ax.scatter(
        true_values,
        predictions,
        s=70,
        alpha=0.8,
        color=VIOLIN_COLOR,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    lo = min(true_values.min(), predictions.min())
    hi = max(true_values.max(), predictions.max())
    pad = 0.08 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, "k--", linewidth=2, alpha=0.8, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(False)

    format_axis_labels(
        ax,
        xlabel=r"PMF-derived $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        ylabel=r"Predicted $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        title=None,
    )
    # The axis labels are long relative to the scatter panels; shrink both so
    # they stay within the panel (both scatter panels share the same column
    # width and height).
    ax.xaxis.label.set_fontsize(29)
    ax.yaxis.label.set_fontsize(29)

    # Stats box mirrors the Supplementary Fig. S10b legend style: a frameon
    # legend carrying the metrics as pure text. n is on its own line (not
    # parenthesised after MAE) for readability.
    # Lead identities are not repeated inside the figure; the caption states
    # them for each scatter panel.
    stats_text = (
        f"$n$ = {len(true_values)}"
        f"\nPearson $r$ = {row['pearson_r']:.2f}"
        f"\nMAE = {row['mae']:.2f} $\\mathrm{{kcal\\cdot mol^{{-1}}}}$"
    )
    ax.legend(
        [Line2D([], [], linestyle="none", marker="")],
        [stats_text],
        loc="lower right", frameon=True,
        fontsize=PUBLICATION_FONTS["annotation"],
        prop={"family": "Times New Roman"},
        handlelength=0, handletextpad=0.2,
    )


def main() -> None:
    """Draw the 2 x 2 panel set and save the figure to manuscript/Figures."""
    plt.rcParams.update(get_publication_style("default"))

    df = load_results()
    print(f"Loaded {len(df)} lead combinations from the lead-sensitivity database")
    best_r_2 = df.loc[df["lead_count"] == 2, "pearson_r"].idxmax()
    best_r_3 = df["pearson_r"].idxmax()
    for label, idx in (("best two-lead by r", best_r_2),
                       ("best three-lead by r", best_r_3)):
        row = df.loc[idx]
        print(f"{label}: {row['lead_names']} "
              f"(r = {row['pearson_r']:.4f}, MAE = {row['mae']:.3f})")

    # Shared column widths: the two distribution panels (a, c) have equal
    # width in the left column, and the two scatter panels (b, d) have equal
    # width in the right column.
    fig, axes = plt.subplots(
        2, 2, figsize=(16.5, 13.5),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    plot_lead_count_distribution(
        axes[0, 0], df, "pearson_r",
        ylabel="External Pearson $r$", best="max",
        baseline=STRICT_LOGO_R, star_best_mae=True,
        legend_loc="lower left", legend_frameon=False)
    plot_best_combination_scatter(axes[0, 1], df.loc[best_r_3])
    plot_lead_count_distribution(
        axes[1, 0], df, "mae",
        ylabel=r"External MAE ($\mathrm{kcal\cdot mol^{-1}}$)",
        best="min", baseline=STRICT_LOGO_MAE, extra_marker=True,
        legend_loc="upper right", legend_frameon=False)
    plot_best_combination_scatter(axes[1, 1], df.loc[best_r_2])

    # Panel letters: lowercase bold (Scientific Reports convention) and larger
    # than the axis-label font, offset clearly to the left of the y-axis label.
    letter_x = {-1: -0.27, 0: -0.31}  # column index -> offset (axes coords)
    for ax, letter in zip(axes.flat, ["a", "b", "c", "d"]):
        col = ax.get_subplotspec().colspan.start
        ax.text(
            letter_x[col - 1],
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=42,
            fontweight="bold",
            fontfamily="Times New Roman",
        )

    plt.tight_layout()
    save_figure(fig, "FigS7_Lead_Sensitivity")


if __name__ == "__main__":
    main()
