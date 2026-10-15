#!/usr/bin/env python3
"""
Generate Supplementary Fig. S8: Five-fold model selection for final deployment.

Layout: 2 panels side by side
  a. Stage-1 screening performance distribution (violin by model family)
  b. Final ensemble OOF prediction (120 samples, predicted vs actual)

Output: manuscript/Figures/FigS8_Deployment_Model_Selection.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Path setup
BASE = Path(__file__).resolve().parents[3]  # figures/manuscript/ -> scripts/figures -> scripts -> OP_unit
sys.path.insert(0, str(BASE / "scripts" / "figures"))

from plot_utils import (
    PUBLICATION_FONTS, ANION_COLORS, STRICT_FILL, apply_publication_style,
    save_publication_figure
)
from public_data import load_deployment_screening, load_ensemble_oof

# Model family grouping
MODEL_FAMILIES = {
    'Linear': ['ridge', 'bayesian_ridge'],
    'SVM': ['svm_rbf', 'svm_linear'],
    'kNN': ['knn'],
    'Decision Tree': ['decision_tree'],
    'Bagging': ['bagging'],
    'Tree Ensemble': ['random_forest', 'extra_trees'],
    'Boosting': ['gradient_boosting', 'xgboost', 'lightgbm'],
    'Neural Net': ['mlp']
}

def get_model_family(model_name):
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return 'Other'

def panel_a_stage1_distribution(ax, df_stage1):
    """Panel a: Stage 1 screening performance distribution (horizontal violin plot by model family)."""
    df_stage1['family'] = df_stage1['model'].apply(get_model_family)

    families = list(MODEL_FAMILIES.keys())
    data_by_family = [df_stage1[df_stage1['family'] == fam]['primary_metric'].values
                      for fam in families]

    # Horizontal violins: one model family per row (y-axis, readable long labels),
    # CV metric on the x-axis. vert=False + showmedians=True keeps the family
    # median as a short horizontal bar inside each violin (the manuscript caption
    # reports median r per family); showextrema=False drops the min/max whiskers.
    positions = range(len(families))
    parts = ax.violinplot(data_by_family, positions=positions, vert=False,
                          widths=0.7, showmedians=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(STRICT_FILL)
        pc.set_edgecolor('black')

    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(1.6)

    # Jittered individual configurations on top of each violin (house style of
    # Fig. S7), so the mass of the data stays visible next to the KDE tails.
    # For a horizontal violin (family on y, metric on x) the metric goes on the
    # x-axis and the jittered family position on the y-axis.
    rng = np.random.default_rng(42)
    for pos, data in zip(positions, data_by_family):
        ax.scatter(data, rng.normal(pos, 0.07, size=len(data)),
                   alpha=0.35, s=28, color=STRICT_FILL, edgecolor='none')

    ax.set_yticks(list(positions))
    ax.set_yticklabels(families, fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['tick_label'])
    ax.set_xlabel('Five-fold CV Pearson r', fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    # Left limit spans the full data range (worst configurations fall below r=0),
    # so the long lower tails of the violins are not clipped at the axis.
    ax.set_xlim([-0.27, 0.95])

    ax.text(-0.18, 1.15, 'a', transform=ax.transAxes, fontsize=40,
            fontweight='bold', va='top', ha='left', fontfamily='Times New Roman')

def panel_c_ensemble_oof(ax, df_oof):
    """Panel c: Ensemble OOF prediction scatter (120 samples)."""
    x = df_oof['actual'].values
    y = df_oof['predicted'].values

    ax.scatter(x, y, s=40, alpha=0.7, edgecolors='black', linewidths=0.5,
               c=ANION_COLORS['A_N'])

    lims = [min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5]
    ax.plot(lims, lims, 'k-', alpha=0.5, linewidth=1.5)

    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error
    r_val, _ = pearsonr(x, y)
    mae_val = mean_absolute_error(x, y)

    # Stats box mirrors the Supplementary Fig. S10b legend style: a frameon
    # legend carrying the coefficients as pure text, so the box has the same
    # frame/shadow/font as S10 rather than a borderless white text patch.
    ax.legend(
        [Line2D([], [], linestyle='none', marker='')],
        [f'Pearson $r$ = {r_val:.3f}\nMAE = {mae_val:.3f} $\\mathrm{{kcal\\cdot mol^{{-1}}}}$'],
        loc='upper left', frameon=True,
        fontsize=PUBLICATION_FONTS['annotation'],
        prop={'family': 'Times New Roman'},
        handlelength=0, handletextpad=0.2,
    )

    ax.set_xlabel('PMF-derived $\\Delta G_\\mathrm{rel}$ ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    ax.set_ylabel('Predicted $\\Delta G_\\mathrm{rel}$ ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    # Matched limits (not set_aspect('equal','box')) so the axes box fills its
    # grid cell and the long xlabel stays inside the figure edge — mirrors the
    # S7 panel-b approach; equal aspect shrinks the box and pushes the label out.
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.text(-0.20, 1.15, 'b', transform=ax.transAxes, fontsize=40,
            fontweight='bold', va='top', ha='left', fontfamily='Times New Roman')

def main():
    apply_publication_style()

    # Load stage 1 data from CSV (all non-VAE representations)
    df_stage1 = load_deployment_screening()
    # Filter non-VAE
    df_stage1 = df_stage1[~df_stage1['representation'].str.contains('VAE', case=False, na=False)]

    # Load ensemble OOF predictions
    df_oof = load_ensemble_oof("oof_predictions")

    # 2 panels side by side: violin (left) + scatter (right, wider for square aspect)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.2, 1]})

    panel_a_stage1_distribution(ax_a, df_stage1)
    panel_c_ensemble_oof(ax_b, df_oof)

    plt.tight_layout(pad=2.0)

    out_path = BASE / "manuscript" / "Figures" / "FigS8_Deployment_Model_Selection.png"
    save_publication_figure(fig, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
