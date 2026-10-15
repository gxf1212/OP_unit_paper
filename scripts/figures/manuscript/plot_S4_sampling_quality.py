"""Create manuscript Supplementary Fig. S4: umbrella-sampling quality.

Three panels assess the sampling quality behind the 120-motif PMF dataset:

- Panel A: umbrella histograms of all 33 windows for one representative
  replicate (hC001-1, the same fixed first-member selection rule as Fig. 3),
  drawn as unfilled outlines on the sequential window colour scale — the visual overlap check.
- Panel B: global window-overlap statistic. For every motif replicate with a
  WHAM histogram file (599 of the 600 motif-replicate combinations; hH037-3
  has no histo.xvg on the simulation server), adjacent-window histograms are
  normalised and their overlap coefficient sum(min(p_i, p_i+1)) is computed;
  the panel shows the distribution of the per-replicate WORST adjacent-pair
  overlap, so the panel quantifies the worst case over the whole library
  rather than a cherry-picked example.
- Panel C: replicate agreement. For each of the 120 motifs the sample
  standard deviation of DeltaG_rel across the five independent repeats is
  computed from data/pmf_profiles and shown as a library-wide distribution (the
  median of this distribution is the 0.59 kcal/mol label uncertainty quoted in
  the main text and Table S3).

Data provenance: histo.xvg / iact.xvg were fetched from the simulation
server into data/sampling_quality/<system>/;
the merged replicate PMFs live in data/pmf_profiles.

Output: manuscript/Figures/FigS4_Sampling_Quality.png
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_data_analysis import DATA_DIR, MANUSCRIPT_FIGURE_DIR, load_molecular_data
from plot_utils import (PUBLICATION_FONTS, get_publication_style,
                        lighten_color, save_publication_figure)

HISTO_DIR = DATA_DIR / 'sampling_quality'
REPRESENTATIVE = 'hC001-1'
OUTPUT_PATH = MANUSCRIPT_FIGURE_DIR / 'FigS4_Sampling_Quality.png'

# Sequential colour ramp for the window histograms: purple (first window) ->
# blue (last window). Replaces viridis, whose yellow/green range read poorly
# on white and printed weakly, while its dark purple end was the clearest.
WINDOW_CMAP = LinearSegmentedColormap.from_list(
    'window_purple_blue', ['#5B2C90', '#1F77B4'])

# How panel (a) draws the 33 window histograms (request 2026-09-10). All three
# styles use the sequential ramp above (purple -> blue, encoding each window's
# position along z); they differ only in how that colour is applied:
#   'outline'           - no fill, one line per window, so the overlapping
#                         histograms stay legible instead of stacking into
#                         solid bands of colour;
#   'light'             - gradient fills in a pale tint of the same ramp
#                         (each colour blended 45% onto white) with a thin
#                         full-strength outline on top;
#   'filled'  (default) - translucent fills at 55% opacity (the original look,
#                         kept on the purple -> blue ramp).
PANEL_A_STYLE = 'filled'


def read_xvg(path: Path) -> Optional[np.ndarray]:
    """Read a plain multi-column xvg file as a float array (None if absent)."""
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    return np.asarray(rows) if rows else None


def panel_a_representative_overlap(ax: plt.Axes,
                                   style: str = PANEL_A_STYLE) -> None:
    """Umbrella histograms of all windows for the representative replicate.

    Each window keeps the sequential purple-to-blue colour (WINDOW_CMAP) that
    encodes its position along z (no colourbar: the x axis already shows z):
      ``'outline'`` (default) - that colour as a line, no fill, so the 33
                                overlapping histograms stay legible rather
                                than stacking into solid bands;
      ``'light'``             - a pale gradient fill of the same colour
                                (blended onto white) with a thin
                                full-strength outline drawn on top;
      ``'filled'``            - the original translucent fills.
    """
    data = read_xvg(HISTO_DIR / REPRESENTATIVE / 'histo.xvg')
    z, counts = data[:, 0], data[:, 1:]
    cmap = WINDOW_CMAP
    norm = plt.Normalize(0, counts.shape[1] - 1)
    for w in range(counts.shape[1]):
        colour = cmap(norm(w))
        if style == 'filled':
            ax.fill_between(z, counts[:, w], color=colour, alpha=0.55,
                            linewidth=0)
        elif style == 'light':
            ax.fill_between(z, counts[:, w],
                            color=lighten_color(colour, 0.45), linewidth=0)
            ax.plot(z, counts[:, w], color=colour, linewidth=0.8, alpha=0.9)
        else:
            ax.plot(z, counts[:, w], color=colour, linewidth=1.2, alpha=0.8)

    ax.set_xlabel('$z$ (nm)', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    # Counts are reported in thousands: the panel reaches ~1e4, so the raw
    # tick text (5000, 10000) is replaced by 0, 2, 4 ... and the scale factor
    # moves into the axis label (request 2026-09-10). The default AutoLocator
    # already lands on multiples of 2000, so no custom locator is needed.
    ax.set_ylabel('Count ($\\times 10^{3}$)',
                  fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e3:g}'))
    ax.set_xlim(z.min(), z.max())
    # Counts cannot be negative: start at 0 rather than letting the default
    # 5% margin open a spurious negative band below the data.
    ax.set_ylim(0, counts.max() * 1.05)
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def adjacent_overlap_coefficients(histo: np.ndarray) -> np.ndarray:
    """Overlap coefficient between each pair of adjacent window histograms.

    Each histogram is normalised to a probability distribution and the
    overlap is sum(min(p_i, p_i+1)) in [0, 1].
    """
    counts = histo[:, 1:]
    totals = counts.sum(axis=0)
    usable = totals > 0
    prob = counts[:, usable] / totals[usable]
    return np.minimum(prob[:, :-1], prob[:, 1:]).sum(axis=0)


def used_replicate_systems() -> list[str]:
    """The 600 motif-replicate combinations actually behind the dataset.

    Mirrors the replicate substitution in ``replicate_delta_g``: hH037
    abandoned its third replicate and substituted a sixth one.
    """
    head = load_molecular_data()
    systems = []
    for motif in head['Name'].astype(str):
        for rep in range(1, 6):
            if (DATA_DIR / 'pmf_profiles' / f'{motif}-{rep}.xvg').exists():
                systems.append(f'{motif}-{rep}')
            else:
                systems.append(f'{motif}-6')
    return systems


def collect_worst_overlaps(systems: list[str]) -> np.ndarray:
    """Worst adjacent-pair overlap per replicate, over the dataset replicates."""
    worst = []
    for system in systems:
        data = read_xvg(HISTO_DIR / system / 'histo.xvg')
        if data is None:
            print(f'  warning: no histo.xvg for {system}')
            continue
        worst.append(adjacent_overlap_coefficients(data).min())
    return np.asarray(worst)


def panel_b_overlap_distribution(ax: plt.Axes, worst: np.ndarray) -> None:
    """Distribution of the worst adjacent-window overlap per replicate.

    The fill is deliberately a pale tint of the default blue: panels B and C
    are global quality statistics that sit beneath panel A's stack of window
    outlines in the composite, and a full-strength bar would out-draw panel A.
    """
    ax.hist(worst, bins=30, color='#B7C9DE',
            edgecolor='#4C72B0', linewidth=0.9)
    median = float(np.median(worst))
    ax.axvline(median, color='black', linestyle='--', linewidth=1.6)
    ax.annotate(f'median = {median:.2f}',
                xy=(median, ax.get_ylim()[1]), xytext=(-6, -8),
                textcoords='offset points', ha='right',
                fontsize=PUBLICATION_FONTS['annotation'],
                fontfamily='Times New Roman', va='top')
    print(f"  worst-overlap: n = {len(worst)}, median = {median:.3f}, "
          f"min = {worst.min():.3f}")

    ax.set_xlabel('Worst adjacent-window overlap',
                  fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel('Replicates', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def replicate_delta_g(motif: str) -> np.ndarray:
    """DeltaG_rel of each replicate: G(1.6 nm) - G(4.0 nm).

    The dataset convention is five replicates per motif. hH037 abandoned its
    third replicate and substituted a sixth one, so a missing replicate in
    1-5 is replaced by rep 6 (the same five profiles behind the hH037 row of
    public reference table). Systems without five usable profiles are dropped.

    The SD uses ddof=1 (sample SD): the label-uncertainty statistics quoted in
    the main text (median 0.59 kcal/mol) and Table S3 use the sample SD across
    the five repeats, and the figure must reproduce those exact numbers.
    """
    values = []
    for rep in range(1, 6):
        path = DATA_DIR / 'pmf_profiles' / f'{motif}-{rep}.xvg'
        if not path.exists():
            path = DATA_DIR / 'pmf_profiles' / f'{motif}-6.xvg'
        data = read_xvg(path)
        if data is None:
            continue
        z, g = data[:, 0], data[:, 1]
        values.append(np.interp(1.6, z, g) - np.interp(4.0, z, g))
    return np.asarray(values)


def collect_replicate_sds() -> np.ndarray:
    """Per-motif SD of DeltaG_rel across the five replicates."""
    head = load_molecular_data()
    motifs = head['Name'].astype(str).tolist()
    sds = []
    for motif in motifs:
        values = replicate_delta_g(motif)
        if len(values) == 5:
            sds.append(values.std(ddof=1))
    return np.asarray(sds)


def panel_c_replicate_sd(ax: plt.Axes, sds: np.ndarray) -> None:
    """Distribution of the per-motif replicate SD of DeltaG_rel.

    Same pale-tint treatment as panel B, keeping B and C a matched pair of
    quiet panels under panel A.
    """
    ax.hist(sds, bins=25, color='#F2CDB0',
            edgecolor='#DD8452', linewidth=0.9)
    median = float(np.median(sds))
    ax.axvline(median, color='black', linestyle='--', linewidth=1.6)
    ax.annotate(f'median = {median:.2f}',
                xy=(median, ax.get_ylim()[1]), xytext=(5, -8),
                textcoords='offset points',
                fontsize=PUBLICATION_FONTS['annotation'],
                fontfamily='Times New Roman', va='top')
    print(f"  replicate SD: n = {len(sds)}, median = {median:.3f}, "
          f"mean = {sds.mean():.3f}, range = {sds.min():.2f}-{sds.max():.2f}")

    ax.set_xlabel(r'SD of $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                  fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel('Motifs', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold')
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def main() -> None:
    plt.rcParams.update(get_publication_style())
    # Panel A spans the full top row; the two global-statistic histograms
    # (B, C) share the second row. The 16-in width matches the other SI
    # composites (e.g. Supplementary Figs S2 and S3) so the shared font sizes keep
    # the same proportion to the panels.
    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    print('Panel A: representative window histograms...')
    panel_a_representative_overlap(ax_a)
    print('Panel B: worst adjacent-window overlap across replicates...')
    panel_b_overlap_distribution(
        ax_b, collect_worst_overlaps(used_replicate_systems()))
    print('Panel C: replicate SD of DeltaG_rel across motifs...')
    panel_c_replicate_sd(ax_c, collect_replicate_sds())

    # Extra vertical padding keeps panel A's x label clear of the C letter.
    fig.tight_layout(h_pad=3.2, w_pad=3.0)
    # Letters are placed in figure coordinates just left of each panel's top
    # left corner, so A and B (same column) are horizontally aligned even
    # though their y-label widths differ. C takes a larger offset because the
    # gap between the two columns is narrower than the left margin.
    letter_dx = {'a': -0.055, 'b': -0.055, 'c': -0.085}
    for letter, ax in zip('abc', [ax_a, ax_b, ax_c]):
        pos = ax.get_position()
        fig.text(pos.x0 + letter_dx[letter], pos.y1 + 0.022, letter,
                 fontsize=PUBLICATION_FONTS['title'], fontweight='bold',
                 fontfamily='Times New Roman', ha='left', va='bottom')

    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, str(OUTPUT_PATH))
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
