"""Shared tile geometry and styling for manuscript Figure 2 panels.

All Figure 2 panels and 2x2 supplementary figures reuse the same fixed-layout
tile: axes box 6.30 x 4.725 in, left margin 1.35 in, bottom margin 2.60 in
(accommodates rotated tick labels + xlabel + legend band), on a 7.8 x 7.475 in
canvas saved at 300 dpi without bbox_inches='tight', so all tiles can be
pasted 1:1 with no rescaling and the effective font size is identical across
all manuscript figures.
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import PUBLICATION_FONTS, get_publication_style

# Shared canvas geometry (inches at 300 dpi)
FIG_W, FIG_H = 7.8, 7.475
AX_RECT = [1.35 / FIG_W, 2.60 / FIG_H, 6.30 / FIG_W, 4.725 / FIG_H]
LEGEND_Y = 0.55 / FIG_H  # legend centre, in the band below the x label
LEGEND_X = (1.35 + 6.30 / 2) / FIG_W  # centre on axes box, not canvas


def make_tile(left_band: float = 0.0):
    """Create an empty Figure 2 tile with the shared fixed axes geometry.

    Parameters
    ----------
    left_band : float
        Extra canvas width (inches) added to the left of the axes box, used by
        the Figure 2b tile to host its vertical R-group legend in the
        otherwise-empty corridor left of the panel. The axes box keeps its
        exact 6.30 x 4.725 in size and its vertical position, so the tile still
        pastes 1:1 (assemble_fig2.py shifts the tile left by this amount to
        keep the axes box in place).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects.
    """
    plt.rcParams.update(get_publication_style())
    fig_w = FIG_W + left_band
    fig = plt.figure(figsize=(fig_w, FIG_H), dpi=300)
    rect = ((left_band + 1.35) / fig_w, 2.60 / FIG_H,
            6.30 / fig_w, 4.725 / FIG_H)
    ax = fig.add_axes(rect)
    return fig, ax


def style_tile(ax: plt.Axes) -> None:
    """Apply the shared Figure 2 spine/tick styling to a tile axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'],
                   pad=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')


# Legacy aliases for backward compatibility
_new_tile = make_tile
_style_axes = style_tile
