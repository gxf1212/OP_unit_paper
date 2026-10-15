"""Assemble the manuscript Figure 2 from its component tiles.

Layout (2x2 grid, reading order follows the Results narrative — marginal
effects first, then the joint landscape, then the conditional linker step):

    (a) Fig2a_DeltaG_Distribution_6subgroups.png  (b) Fig2b_Rgroup_Box.png
    (c) Fig2c_Mean_dGrel_Heatmap_8x6.png          (d) Fig2d_dG_step_boxplot.png

Panels (b) and (c) were swapped on 2026-09-07 (heatmap moved from top-right to
bottom-left) so that the top row shows the two single-factor distributions
(terminal chemistry, R group) before the joint 48-combination heatmap and the
linker-length conditional analysis below.

All four tiles are produced at fixed pixel sizes (300 dpi, exact figsize,
saved without bbox_inches='tight') and are pasted 1:1 with no rescaling, so
the effective font size is identical in every panel. The heatmap tile is wider
than the other three (2880 px vs 2340 px), so the column boundary is derived
from the wider left-column tile; the axes boxes are all 6.30 x 4.725 in and
the tiles therefore align across columns at their left edges. All four tiles
are standard-width and pasted 1:1; panel b no longer carries an R-group
colour legend (its eight R groups are labelled on the x-axis, and the colours
are used only for consistency with Fig. 3 and Supplementary Fig. S2, as noted
in the Figure 2 caption). Panel letters are bold lowercase Times New Roman
drawn here (the component tiles themselves are letter-free), matching
the (a)-(d) lettering used in the manuscript figure legend and the lowercase
panel letters required by Scientific Reports.

Output: manuscript/Figures/Fig2.png
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[3]
FIG_DIR = BASE / 'manuscript' / 'Figures'
OUTPUT_PATH = FIG_DIR / 'Fig2.png'

TILE_A = 'Fig2a_DeltaG_Distribution_6subgroups.png'
TILE_B = 'Fig2b_Rgroup_Box.png'
TILE_C = 'Fig2c_Mean_dGrel_Heatmap_8x6.png'
TILE_D = 'Fig2d_dG_step_boxplot.png'

PAD = 60          # outer white margin (px)
TOP = 260         # margin above row 1 for the a/b letters (px)
GAP_COL = 150     # horizontal gap between the tiles (px)
GAP_ROW = 0       # rows touch directly; c/d letters overlap the row-1
                  # tiles' empty bottom margins

LETTER_FONTSIZE = 40
LETTER_FAMILY = 'Times New Roman'


def _load(name):
    """Load one tile unchanged (tiles are assembled 1:1, no rescaling)."""
    return mpimg.imread(str(FIG_DIR / name))


def assemble():
    """Stitch the component tiles into the 2x2 Figure 2 canvas."""
    tile_a = _load(TILE_A)
    tile_b = _load(TILE_B)
    tile_c = _load(TILE_C)
    tile_d = _load(TILE_D)

    # Column width driven by the wider tile in that column (heatmap in the
    # left column), row height by the taller tile in that row.
    col1_x = PAD
    col2_x = PAD + max(tile_a.shape[1], tile_c.shape[1]) + GAP_COL
    row1_y = TOP
    row1_h = max(tile_a.shape[0], tile_b.shape[0])
    row2_y = row1_y + row1_h + GAP_ROW
    row2_h = max(tile_c.shape[0], tile_d.shape[0])

    # All tiles are standard-width, so no legend band shift is needed: panel b
    # pastes at the same column origin as panel d and aligns at its left edge.
    band_b = tile_b.shape[1] - tile_a.shape[1]
    tile_b_x = col2_x - band_b

    tiles = [(tile_a, col1_x, row1_y), (tile_b, tile_b_x, row1_y),
             (tile_c, col1_x, row2_y), (tile_d, col2_x, row2_y)]
    width = max(x + t.shape[1] for t, x, _ in tiles) + PAD
    height = row2_y + row2_h + PAD
    canvas = np.ones((height, width, 4), dtype=float)

    for tile, x, y in tiles:
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile

    dpi = 300
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, interpolation='antialiased')
    ax.axis('off')

    # Letters sit in the band above their tile, 30 px clear of the tile edge;
    # 40 pt at 300 dpi is ~167 px tall, hence the 260 px letter bands.
    letter_h = round(LETTER_FONTSIZE / 72 * dpi)
    y_top_row1 = row1_y - 30 - letter_h
    y_top_row2 = row2_y - 30 - letter_h
    # Letters align per column (a/b at col1_x, c/d at col2_x); b stays at
    # col2_x even though the tile is pasted further left for its legend band.
    for letter, x, y in [('a', col1_x, y_top_row1), ('b', col2_x, y_top_row1),
                         ('c', col1_x, y_top_row2), ('d', col2_x, y_top_row2)]:
        ax.text(x + 10, y, letter, fontsize=LETTER_FONTSIZE,
                fontweight='bold', family=LETTER_FAMILY, ha='left', va='top')

    fig.savefig(OUTPUT_PATH, dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH} ({width} x {height} px)")


if __name__ == '__main__':
    assemble()
