"""Shared 2x2 assembler that reuses the exact Figure 2 tile + assembly style.

Each panel is built as a letter-free Figure 2 tile via ``_fig2_tile_base`` /
``make_tile`` (axes box 6.30 x 4.725 in, full four-spine frame, Times New
Roman, axis labels at PUBLICATION_FONTS axis_label=32 / tick_label=26), then
the four tiles are stitched into a 2x2 supplementary figure using the *same*
PAD / TOP / panel-letter geometry as ``assemble_fig2.py``.

Two adjustments keep the grid tight (2026-09-04):

1. Each tile reserves a 2.60 in bottom margin for tick labels + x label, and
   the lower half of that margin stays white. The assembler trims each
   rendered tile just below its lowest non-white pixel row, so rows sit
   close together instead of showing a large empty band.
2. The shared legend is a real ``fig.legend`` with standard handles
   (``patch`` -> filled square with black edge, ``line`` -> circle marker
   with black edge), matching the in-tile legend of Fig. 2c. Its band
   position is selected by ``legend_position``: ``'top'`` (historical
   default) draws it in the ``TOP`` band above row 1, ``'bottom'`` draws it
   in the reserved strip below the grid (used by Fig. S2 / S3).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys_path = str(Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, sys_path)

from _fig2_tile_base import make_tile, style_tile
from plot_utils import PUBLICATION_FONTS

DPI = 300
PAD = 60          # outer white margin (px) -- mirrors assemble_fig2
TOP = 360         # band above row 1 for the A/B letters and top shared legend (px)
GAP_COL = 150     # horizontal gap between the two columns (px)
LETTER_FONTSIZE = 40
LETTER_FAMILY = 'Times New Roman'
# Row gap: band for the C/D letters between the trimmed rows (letter height
# ~167 px at 40 pt/300 dpi, 30 px clear of both tiles).
GAP_ROW = 30 + round(LETTER_FONTSIZE / 72 * DPI) + 20
# White margin kept below the lowest content pixel when trimming tiles.
TRIM_MARGIN = 40
# Legend band height below the grid (px), incl. the 15 px gap to the tiles.
LEGEND_BAND = 15 + 130


# --- tile helpers (re-export the Figure 2 tile machinery) -------------------
# Already imported above: make_tile and style_tile from _fig2_tile_base


def _render(fig):
    """Save a tile figure 1:1 (no tight bbox) and return its pixel array."""
    tmp = Path(tempfile.mkdtemp()) / "tile.png"
    fig.savefig(tmp, dpi=DPI, bbox_inches=None, facecolor='white',
                edgecolor='none')
    plt.close(fig)
    return mpimg.imread(tmp)


def _trim_bottom(arrs):
    """Trim each tile just below its lowest non-white pixel row.

    All tiles share the same canvas, so the deepest content row across the
    four tiles defines one common crop line (keeps the tiles equal-sized for
    1:1 pasting).
    """
    bottom = 0
    for a in arrs:
        nonwhite = (a[..., :3] < 0.98).any(axis=2) & (a[..., 3] > 0.1)
        rows = np.where(nonwhite)[0]
        if len(rows):
            bottom = max(bottom, rows[-1])
    crop_h = min(arrs[0].shape[0], bottom + 1 + TRIM_MARGIN)
    return [a[:crop_h] for a in arrs], crop_h


def _build_handles(legend):
    """Translate (kind, colour, label) entries into legend handles."""
    handles = []
    for kind, color, label in legend:
        if kind == 'patch':
            handles.append(Patch(facecolor=color, edgecolor='black',
                                 label=label))
        else:  # 'line' -> circle marker, matching the point style
            handles.append(Line2D([], [], marker='o', linestyle='None',
                                  markersize=11, markerfacecolor=color,
                                  markeredgecolor='black', label=label))
    return handles


def assemble_2x2(tile_figs, letters, legend, out_path,
                 legend_position='top') -> None:
    """Stitch four letter-free tiles into a 2x2 supplementary figure.

    Parameters
    ----------
    tile_figs : list[matplotlib.figure.Figure]
        Four Figure 2 tiles, reading order (row 1 L->R, row 2 L->R). All must
        share the same pixel size.
    letters : list[str]
        Four uppercase panel letters, same reading order.
    legend : list[(str, color, str)]
        ``(kind, color, label)`` entries; ``kind`` is ``'patch'`` (filled
        square) or ``'line'`` (circle marker). Drawn as one shared band.
    out_path : str | Path
        Destination .png path.
    legend_position : {'top', 'bottom'}, default 'top'
        Where the shared legend band sits. ``'top'`` reproduces the original
        layout (legend in the ``TOP`` band above row 1); ``'bottom'`` moves it
        into the reserved strip below the 2x2 grid, clear of the panels.
    """
    arrs = []
    W = H = None
    for fig in tile_figs:
        a = _render(fig)
        if W is None:
            W, H = a.shape[1], a.shape[0]
        if a.shape[1] != W or a.shape[0] != H:
            raise ValueError("all four tiles must share the same pixel size")
        arrs.append(a)
    arrs, H = _trim_bottom(arrs)

    content_h = TOP + H + GAP_ROW + H
    width = PAD + W + GAP_COL + W + PAD
    # A bottom legend needs a taller strip than the (usually blank) one kept
    # under the grid in the top-legend layout, so the 22 pt handles clear both
    # the tiles above and the page edge below.
    legend_band = 180 if legend_position == 'bottom' else LEGEND_BAND
    height = content_h + legend_band

    canvas = np.ones((height, width, 4), dtype=float)
    cell_x = [PAD, PAD + W + GAP_COL]
    cell_y = [TOP, TOP + H + GAP_ROW]
    pos = [(cell_x[0], cell_y[0]), (cell_x[1], cell_y[0]),
           (cell_x[0], cell_y[1]), (cell_x[1], cell_y[1])]
    for a, (x, y) in zip(arrs, pos):
        canvas[y:y + H, x:x + W] = a

    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, interpolation='antialiased')
    ax.axis('off')

    # Panel letters: same band/offset as assemble_fig2.py.
    letter_h = round(LETTER_FONTSIZE / 72 * DPI)
    y_top_row1 = TOP - 30 - letter_h
    y_top_row2 = TOP + H + GAP_ROW - 30 - letter_h
    row_y = [y_top_row1, y_top_row1, y_top_row2, y_top_row2]
    for letter, (x, y), yt in zip(letters, pos, row_y):
        ax.text(x + 10, yt, letter, fontsize=LETTER_FONTSIZE,
                fontweight='bold', family=LETTER_FAMILY, ha='left', va='top')

    # Shared legend band (fig.legend, Fig. 2c style): 'top' reproduces the
    # original layout above row 1; 'bottom' centres it in the strip below the
    # grid, clear of the panels.
    if legend:
        legend_y = ((legend_band / 2) / height if legend_position == 'bottom'
                    else (height - 80) / height)
        fig.legend(handles=_build_handles(legend), loc='center',
                   ncol=len(legend), frameon=False, columnspacing=1.2,
                   handlelength=1.2, handletextpad=0.5,
                   bbox_to_anchor=(0.5, legend_y),
                   prop={'family': 'Times New Roman',
                         'size': PUBLICATION_FONTS['legend']})

    fig.savefig(out_path, dpi=DPI, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({width} x {height} px)")
