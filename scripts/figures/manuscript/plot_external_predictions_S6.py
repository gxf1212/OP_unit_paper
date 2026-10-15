#!/usr/bin/env python3
"""Manuscript Supplementary Fig. S6: external predictions under both LOGO settings.

A symmetric 2 x 5 grid of predicted-versus-PMF-derived scatter plots:

    top row    -- the five Strict LOGO folds (held-out class fully absent
                  from model development)
    bottom row -- the five few-shot LOGO folds (one A/N motif or two motifs
                  of each other class added to training before the complete
                  model-selection procedure was repeated)

Each column corresponds to the same target terminal-chemistry class. The
per-sample predictions come from the retrained final models
(``data/logo_external_predictions.csv``),
so every panel shows exactly the predictions of the model behind the
corresponding Table 1 row.

Notation (user decision 2026-09-02): ML/LOGO figures use the five merged
classes A/N, C, H, P, S (the slash pairs the hA/OH- and hN/O- subgroups and
is NOT a subscript); no chemical formulas inside the figure — the caption
decodes the codes.

Output: manuscript/Figures/FigS6_External_Predictions.png
(shared common_figures and patent figures are not touched).

Usage:
    python plot_external_predictions_S5.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_utils import (
    PUBLICATION_FONTS,
    save_publication_figure,
)
from public_data import load_logo_external_predictions

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    BASE_DIR / 'manuscript' / 'Figures' / 'FigS6_External_Predictions.png'
)

CLASS_ORDER = ['A_N', 'C', 'H', 'P', 'S']
CLASS_LABELS = {'A_N': 'A/N', 'C': 'C', 'H': 'H', 'P': 'P', 'S': 'S'}

# Row order mirrors the caption: strict on top, few-shot below. Colours match
# the strict/few-shot palette of Figure 5 so the two figures read as a pair.
ROWS = [
    ('strict', 'Strict LOGO', '#1f77b4'),
    ('improved', 'Few-shot LOGO', '#ff7f0e'),
]

STATS_FONTSIZE = 16


def load_fold_predictions(logo_type: str, fold: str) -> dict:
    """Load one public long-format LOGO prediction subset."""
    df = load_logo_external_predictions()
    df = df[(df['setting'] == logo_type) & (df['held_out_class'] == fold)]
    if df.empty:
        raise ValueError(f'No public predictions for {logo_type}/{fold}')
    y_true = df['y_true'].to_numpy(dtype=float)
    y_pred = df['y_pred'].to_numpy(dtype=float)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'r': float(np.corrcoef(y_true, y_pred)[0, 1]),
        'mae': float(np.abs(y_true - y_pred).mean()),
        'n': len(y_true),
    }


def draw_panel(ax, data: dict, color: str) -> None:
    """Draw one predicted-versus-measured panel with identity line and stats."""
    y, p = data['y_true'], data['y_pred']
    ax.scatter(y, p, s=70, alpha=0.8, color=color,
               edgecolor='black', linewidth=0.6, zorder=3)

    lo = min(y.min(), p.min())
    hi = max(y.max(), p.max())
    pad = 0.08 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, 'k--', linewidth=1.6, alpha=0.8, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    # Explicit linewidth: the shared style sets grid.linewidth=0, which turns
    # dashed grid lines into an all-zero dash pattern and breaks drawing.
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'] - 6)

    # Stats box mirrors the S6 panel-B styling: square white box, thin grey
    # edge, lower-right corner so it never covers the diagonal cloud.
    stats_text = (
        f"$r$ = {data['r']:.2f}"
        f"\nMAE = {data['mae']:.2f}"
        f"\n$n$ = {data['n']}"
    )
    ax.text(
        0.96, 0.04, stats_text, transform=ax.transAxes,
        va='bottom', ha='right', fontsize=STATS_FONTSIZE,
        fontfamily='Times New Roman',
        bbox={'boxstyle': 'square', 'facecolor': 'white', 'edgecolor': '0.8',
              'linewidth': 1.0, 'alpha': 0.8},
    )


def create_s5_figure() -> tuple[Figure, dict]:
    """Build the 2 x 5 grid; return the per-panel metrics for reporting."""
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(2, 5, figsize=(24, 10.5))
    metrics = {}

    for row_idx, (logo_type, row_label, color) in enumerate(ROWS):
        for col_idx, fold in enumerate(CLASS_ORDER):
            ax = axes[row_idx, col_idx]
            data = load_fold_predictions(logo_type, fold)
            metrics[(logo_type, fold)] = data
            draw_panel(ax, data, color)

            if row_idx == 0:
                # Column header: the shared target-class code.
                ax.set_title(CLASS_LABELS[fold],
                             fontsize=PUBLICATION_FONTS['subtitle'],
                             fontweight='bold', fontfamily='Times New Roman',
                             pad=12)
            if col_idx == 0:
                # One ylabel per row (leftmost panel only): all ten panels
                # share identical axes ranges, so the y label is not repeated.
                ax.set_ylabel(r'Predicted $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                              fontsize=PUBLICATION_FONTS['axis_label'] - 6,
                              fontfamily='Times New Roman')

        # Row label outside the left edge; bbox_inches='tight' keeps it.
        fig.text(0.004, 0.72 - row_idx * 0.46, row_label,
                 rotation=90, va='center', ha='center',
                 fontsize=PUBLICATION_FONTS['axis_label'],
                 fontweight='bold', fontfamily='Times New Roman')

    # All panels plot the same quantities on identity-scaled axes, so a
    # single shared xlabel centred under the grid replaces ten copies.
    # Full axis_label size + bold to match the row/ylabel emphasis.
    fig.text(0.5, -0.018, r'PMF-derived $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
             ha='center', va='bottom',
             fontsize=PUBLICATION_FONTS['axis_label'],
             fontweight='bold', fontfamily='Times New Roman')

    fig.subplots_adjust(wspace=0.30, hspace=0.18)
    return fig, metrics


def main() -> None:
    """Draw the 10 external-prediction panels and save Supplementary Fig. S5."""
    fig, metrics = create_s5_figure()
    print("Per-panel external-test metrics (from retrained predictions.json):")
    for (logo_type, fold), data in metrics.items():
        print(f"  {logo_type:8s} {fold:3s}  r = {data['r']:.3f}  "
              f"MAE = {data['mae']:.3f}  n = {data['n']}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, str(OUTPUT_PATH))
    print(f"  saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
