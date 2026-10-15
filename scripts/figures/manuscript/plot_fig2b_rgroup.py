"""Create manuscript Figure 2b tile: R-group effect on membrane affinity.

Panel b: PMF-derived deltaG_rel grouped by the eight R groups (linear chains
first, cyclic rings second), with every individual motif shown as a scatter
point coloured by the manuscript eight-colour R-group palette defined in
``plot_data_analysis.R_GROUP_COLORS``.

Fixed-layout tile: a standard Figure 2 tile (axes box 6.30 x 4.725 in, left
margin 1.35 in, bottom 2.60 in on a 7.8 x 7.475 in canvas, saved without
bbox_inches='tight'); assemble_fig2.py pastes it 1:1 with the other three
panels so the four panels keep uniform effective font sizes. The R-group
colour legend is intentionally omitted: the eight R groups are already
labelled on the x-axis, and the colours are used only for visual consistency
with Fig. 3 and Supplementary Fig. S2 (see the Figure 2 caption). Panel
letter is added by assemble_fig2.py.

Output: manuscript/Figures/Fig2b_Rgroup_Box.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_data_analysis import (
    MANUSCRIPT_FIGURE_DIR,
    R_GROUP_COLORS,
    R_GROUP_ORDER,
    R_GROUP_SHORT_LABELS,
    classify_r_group,
    load_molecular_data,
)
from plot_utils import (
    PUBLICATION_FONTS,
    center_rotated_xticklabels,
)
from _fig2_tile_base import make_tile, style_tile

OUTPUT_NAME = 'Fig2b_Rgroup_Box.png'

# Shared scatter convention: fully saturated fill with a thin black outline
SCATTER_KWARGS = {
    'marker': 'o',
    'alpha': 0.8,
    'edgecolor': 'black',
    'linewidth': 0.6,
    's': 55,
    'zorder': 3,
}


def _prepare_dataframe():
    """Load the public reference table and attach the R-group column."""
    df = load_molecular_data()
    df = df.copy()
    df['r_group'] = df['Name'].apply(classify_r_group)
    return df


def create_fig2b(df):
    """Panel b: deltaG_rel box + scatter for each of the eight R groups."""
    fig, ax = make_tile()

    data_by_group = [
        df.loc[df['r_group'] == group, 'dG_rel_kcal_per_mol'].to_numpy()
        for group in R_GROUP_ORDER
    ]
    for group, values in zip(R_GROUP_ORDER, data_by_group):
        print(f"  {group}: n = {len(values)}")

    positions = np.arange(1, len(R_GROUP_ORDER) + 1)
    bp = ax.boxplot(
        data_by_group,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
    )
    for patch, group in zip(bp['boxes'], R_GROUP_ORDER):
        patch.set_facecolor(R_GROUP_COLORS[group])
        patch.set_edgecolor('black')
        patch.set_alpha(0.55)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.6)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='white',
                  markeredgecolor='black', markersize=6)

    rng = np.random.default_rng(42)
    for idx, (values, group) in enumerate(zip(data_by_group, R_GROUP_ORDER), 1):
        ax.scatter(idx, values.mean(), marker='D', color='darkred',
                   s=60, zorder=5)
        jitter = rng.normal(0, 0.05, size=len(values))
        ax.scatter(np.full_like(values, idx, dtype=float) + jitter, values,
                   c=R_GROUP_COLORS[group], **SCATTER_KWARGS)

    # Rotated labels with their geometric centre exactly on each box's tick.
    center_rotated_xticklabels(
        ax,
        [R_GROUP_SHORT_LABELS[g] for g in R_GROUP_ORDER],
        ticks=list(positions),
        rotation=30,
        pad=9,
    )
    ax.set_xlabel('R Group', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=14)
    ax.set_ylabel(r'$\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                  fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=12)
    style_tile(ax)
    ax.grid(False)
    ax.set_axisbelow(True)
    return fig


def _save_tile(fig, output_name: str) -> None:
    """Save a fixed-layout tile at its exact pixel size (no tight bbox)."""
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MANUSCRIPT_FIGURE_DIR / output_name
    fig.savefig(str(save_path), dpi=300, bbox_inches=None,
                facecolor='white', edgecolor='none')
    print(f"  saved: {save_path}")


def main():
    df = _prepare_dataframe()
    fig_b = create_fig2b(df)
    _save_tile(fig_b, OUTPUT_NAME)


if __name__ == '__main__':
    main()
