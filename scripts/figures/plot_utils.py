"""Plot helper utilities shared across OP_unit figure scripts.

This module merges the former `publication_utils.py` (generic PRISM-mirrored
style kernel) and `plot_utils.py` (OP_unit project glue) into a single import
point. The style helpers mirror `/data/gxf1212/work/PRISM/prism/analysis/plots`
so OP_unit figures share identical typography, legend formatting, and layout
defaults; the OP_unit layer adds anion-class colors/labels and the
destination-routing for `common_figures` / `report/figures` / `patent/figures`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fonts, panel sizes and palettes (generic style kernel)
# ---------------------------------------------------------------------------

# Publication-quality font sizes for subfigures/panels.
PUBLICATION_FONTS = {
    'title': 36,
    'subtitle': 32,
    'axis_label': 32,
    'tick_label': 26,
    'legend': 22,
    'annotation': 24,
    'colorbar': 28,
    'value_text': 20,
    'bar_annotation': 16,
}

# Standard panel sizes shared with the PRISM plotting utilities.
STANDARD_PANEL_SIZES = {
    'single': (8, 6),
    'horizontal': (12, 6),
    'vertical': (8, 12),
    'quad': (12, 10),
    'wide': (16, 6),
    'tall': (8, 10),
    'distribution': (12, 8),
}

# Color palettes for scientific publications.
PUBLICATION_COLORS = {
    'default': ['#4A90E2', '#F5A623', '#7ED321', '#D0021B', '#9013FE'],
    'nature': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'science': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161'],
    'accessible': ['#0173B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442'],
    'example': ['#A8DADC', '#D4A574', '#E9C46A', '#457B9D', '#F1FAEE'],
}

# Fixed mapping for the five anion classes used in OP_unit.
ANION_COLORS = {
    'A_N': '#1f77b4',
    'C': '#ff7f0e',
    'H': '#2ca02c',
    'P': '#d62728',
    'S': '#9467bd',
}
PUBLICATION_COLORS['anion'] = list(ANION_COLORS.values())

# ---------------------------------------------------------------------------
# Manuscript ML/LOGO palette (Figure 4 violins, Figure 5 bars)
# ---------------------------------------------------------------------------

# Strict / few-shot LOGO base hues (same as the scatter points of the
# Supplementary Figures S5/S6).
STRICT_COLOR = '#1f77b4'
FEWSHOT_COLOR = '#ff7f0e'

# Large-area fills at full saturation read as too heavy on the page, so the
# violin/bar fills are the base hues composited onto white at FILL_ALPHA
# (softer than the 0.8 of the scatter points). Black outlines stay fully
# opaque because the blend is baked into the face colour instead of using
# Patch.set_alpha, which would also fade the edges.
FILL_ALPHA = 0.50


def lighten_color(color: str, alpha: float = FILL_ALPHA) -> str:
    """Composite a colour onto white at opacity ``alpha`` and return the hex.

    Emulates ``alpha=`` transparency over a white canvas without fading edge
    colours, so fills and legend patches can share one exact light value.

    Args:
        color: Base colour parseable by ``matplotlib.colors.to_rgb``.
        alpha: Opacity of the base colour over the white background.

    Returns:
        Hex string of the blended colour.
    """
    red, green, blue = mcolors.to_rgb(color)
    blended = tuple(channel * alpha + (1.0 - alpha) for channel in (red, green, blue))
    return mcolors.to_hex(blended)


STRICT_FILL = lighten_color(STRICT_COLOR)
FEWSHOT_FILL = lighten_color(FEWSHOT_COLOR)

# Chemical formula labels with proper superscripts for publication figures.
ANION_LABELS = {
    "A_N": r"$\mathrm{O^{-}}$",
    "C": r"$\mathrm{CO_{2}^{-}}$",
    "H": r"$\mathrm{SO_{3}^{-}}$",
    "P": r"$\mathrm{OPO_{2}^{-}OMe}$",
    "S": r"$\mathrm{OSO_{3}^{-}}$",
}

# Full chemical names for text descriptions (not for figure labels).
ANION_FULL_NAMES = {
    "A_N": r"Oxide ($\mathrm{O^{-}}$)",
    "C": r"Carboxylate ($\mathrm{CO_{2}^{-}}$)",
    "H": r"Sulfonate ($\mathrm{SO_{3}^{-}}$)",
    "P": r"Phosphate ester ($\mathrm{OPO_{2}^{-}OMe}$)",
    "S": r"Sulfone oxide ($\mathrm{OSO_{3}^{-}}$)",
}

# ---------------------------------------------------------------------------
# Output routing (OP_unit project layer)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_FIGURE_DIR = BASE_DIR / "report" / "figures"
COMMON_FIGURE_DIR = BASE_DIR / "common_figures"
PATENT_FIGURE_DIR = BASE_DIR / "patent" / "figures"
MANUSCRIPT_FIGURE_DIR = BASE_DIR / "manuscript" / "Figures"

DESTINATION_PATHS = {
    "report": REPORT_FIGURE_DIR,
    "common": COMMON_FIGURE_DIR,
    "patent": PATENT_FIGURE_DIR,
    "manuscript": MANUSCRIPT_FIGURE_DIR,
}

FIGURE_DESTINATIONS = {
    "Fig1_3_Combined_Data_Analysis": "common",  # Combined Fig 1.3 (was Fig1_3 + Fig1_4)
    "Fig1_4_PMF_Curves": "common",  # Renamed from Fig1_5
    "Fig1_14_Prediction_Scatter": "common",
    "Fig1_11_Highthroughput_Prediction": "common",  # High-throughput prediction heatmap
    "Fig1_10_Lead_Sweep": "common",  # Lead molecule sensitivity analysis
    "FigS1_Head_vs_3mer_dG_rel": "manuscript",  # Manuscript supplementary: head vs 3-mer ΔG
    "FigS6_Lead_Sensitivity": "manuscript",  # Old name of FigS7 before renumbering
    "FigS7_Lead_Sensitivity": "manuscript",  # Manuscript supplementary: lead-motif sensitivity (696 combos)
}

DEFAULT_DESTINATION = "report"


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def get_publication_style(style_type: str = 'default') -> dict[str, Any]:
    """Return matplotlib rcParams tuned for subfigure-style panels.

    CRITICAL: This function sets ALL text elements to use the same font family
    and sizes, ensuring complete consistency across legends, colorbars,
    annotations, titles, labels, and any other text rendered by matplotlib.
    """
    base_style = {
        # ============================================================
        # FONT FAMILY - Applied to EVERYTHING
        # ============================================================
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
        'font.sans-serif': ['Times New Roman', 'Arial', 'DejaVu Sans', 'sans-serif'],
        'font.monospace': ['Times New Roman', 'Courier New', 'DejaVu Sans Mono', 'monospace'],
        'font.cursive': ['Times New Roman', 'Comic Sans MS', 'cursive'],
        'font.fantasy': ['Times New Roman', 'Impact', 'fantasy'],

        # ============================================================
        # FONT SIZES - Every text element explicitly defined
        # ============================================================
        'font.size': PUBLICATION_FONTS['tick_label'],          # Global default
        'axes.titlesize': PUBLICATION_FONTS['title'],          # Subplot titles
        'axes.labelsize': PUBLICATION_FONTS['axis_label'],     # X/Y axis labels
        'xtick.labelsize': PUBLICATION_FONTS['tick_label'],    # X-axis tick labels
        'ytick.labelsize': PUBLICATION_FONTS['tick_label'],    # Y-axis tick labels
        'legend.fontsize': PUBLICATION_FONTS['legend'],        # Legend text
        'legend.title_fontsize': PUBLICATION_FONTS['legend'],  # Legend title
        'figure.titlesize': PUBLICATION_FONTS['title'],        # Figure suptitle
        'figure.labelsize': PUBLICATION_FONTS['axis_label'],   # Figure labels

        # ============================================================
        # FONT WEIGHTS
        # ============================================================
        'axes.labelweight': 'bold',      # Axis labels bold
        'axes.titleweight': 'bold',      # Subplot titles bold
        'figure.titleweight': 'bold',    # Figure title bold

        # ============================================================
        # DPI and SAVE SETTINGS
        # ============================================================
        'figure.dpi': 100,
        'savefig.dpi': 200,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',

        # ============================================================
        # GRID SETTINGS (NO GRID by default)
        # ============================================================
        'grid.alpha': 0.0,
        'grid.linewidth': 0.0,
        'axes.grid': False,
        'axes.axisbelow': True,

        # ============================================================
        # AXES STYLING
        # ============================================================
        'axes.linewidth': 2.0,
        'axes.edgecolor': 'black',
        'axes.labelpad': 8,
        'axes.facecolor': 'white',

        # ============================================================
        # TICK STYLING
        # ============================================================
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # ============================================================
        # LEGEND STYLING
        # ============================================================
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.5,
        'legend.handlelength': 2.0,
        'legend.handleheight': 0.7,
        'legend.handletextpad': 0.8,
        'legend.borderaxespad': 0.5,
        'legend.columnspacing': 2.0,

        # ============================================================
        # ERROR BAR AND LINE STYLING
        # ============================================================
        'errorbar.capsize': 12,
        'lines.linewidth': 4.0,
        'lines.markersize': 12,

        # ============================================================
        # FIGURE BACKGROUND
        # ============================================================
        'figure.facecolor': 'white',

        # ============================================================
        # TEXT RENDERING (ensures all text uses same settings)
        # ============================================================
        'text.color': 'black',
        'text.usetex': False,  # Don't use LaTeX (keeps font consistent)
        # Mathtext configuration so chemical formulas stay upright Times New Roman
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.tt': 'Times New Roman',
        'mathtext.cal': 'Times New Roman:italic',
        'mathtext.default': 'rm',  # Render chemistry labels in roman (non-italic)
    }

    return base_style


def get_standard_figsize(panel_type: str = 'single') -> tuple[float, float]:
    """Return canonical figure sizes for the requested panel layout."""
    if panel_type in STANDARD_PANEL_SIZES:
        return STANDARD_PANEL_SIZES[panel_type]

    logger.warning("Unknown panel type '%s', falling back to 'single'", panel_type)
    return STANDARD_PANEL_SIZES['single']


def validate_figure_size(figsize: tuple[float, float], dpi: int = 150) -> tuple[float, float]:
    """Ensure figure dimensions remain within Matplotlib's pixel limits."""
    max_pixels = 32000
    width_px = figsize[0] * dpi
    height_px = figsize[1] * dpi

    if width_px > max_pixels or height_px > max_pixels:
        scale = min(max_pixels / width_px, max_pixels / height_px)
        new_figsize = (figsize[0] * scale, figsize[1] * scale)
        logger.warning(
            "Figure size %s at %sdpi exceeds safe window; scaled to %s",
            figsize,
            dpi,
            new_figsize,
        )
        return new_figsize

    return figsize


def get_color_palette(palette: str = 'default', n_colors: int = 5) -> list[str]:
    """Return publication-safe palette with optional extension."""
    if palette not in PUBLICATION_COLORS:
        palette = 'default'

    colors = PUBLICATION_COLORS[palette]

    if n_colors > len(colors):
        extra_colors = plt.cm.Set3(np.linspace(0, 1, n_colors - len(colors)))
        colors = colors + [mcolors.to_hex(c) for c in extra_colors]

    return colors[:n_colors]


def setup_publication_figure(
    figsize: tuple[float, float] | None = None,
    panel_type: str = 'single',
    style_type: str = 'default',
    title: str = '',
):
    """Create a figure with PRISM publication rcParams applied locally."""
    if figsize is None:
        figsize = get_standard_figsize(panel_type)

    plt.style.use('default')
    with plt.rc_context(get_publication_style(style_type)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)

    return fig, ax


def setup_plot_style(figsize=(10, 8), dpi=300, panel_type="single", style="default"):
    """Setup publication-quality plot styling (project-level entry point)."""
    return setup_publication_figure(
        figsize=figsize, panel_type=panel_type, style_type=style
    )


def apply_publication_style(style_type: str = 'default') -> None:
    """Set matplotlib rcParams globally for downstream plotting sessions."""
    plt.rcParams.update(get_publication_style(style_type))
    logger.info("Applied %s publication style globally", style_type)


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------

def fix_rotated_labels(
    ax,
    labels,
    positions=None,
    rotation: float = 45,
    ha: str = 'center',
    va: str = 'top',
    fontsize: int | None = None,
    manual_alignment: bool = True,
):
    """Align rotated tick labels for publication-quality panel layouts."""
    if positions is None:
        positions = ax.get_xticks()

    if fontsize is None:
        fontsize = PUBLICATION_FONTS['tick_label']

    ax.set_xticks(positions)

    if manual_alignment and rotation != 0:
        ax.set_xticklabels(
            labels,
            rotation=rotation,
            ha=ha,
            va=va,
            fontsize=fontsize,
            weight='normal',
        )
        ax.tick_params(axis='x', which='major', pad=8, labelsize=fontsize)
    else:
        ax.set_xticklabels(
            labels,
            rotation=rotation,
            ha=ha,
            va=va,
            fontsize=fontsize,
            weight='normal',
        )


def center_rotated_xticklabels(
    ax,
    labels,
    ticks=None,
    rotation: float = 30,
    fontsize: int | None = None,
    pad: float = 10,
):
    """Place rotated x tick labels with their geometric centre on the tick.

    ``rotation_mode='anchor'`` keeps the anchor point fixed while the text
    rotates, but with ``ha='center', va='top'`` the glyph centre is still
    displaced from the tick by ``(label height / 2) * sin(rotation)``. This
    helper measures each rendered label with the figure renderer and shifts it
    in data coordinates so the centre of the text lands exactly on the tick,
    as required for multi-column heatmaps and boxplots where every label must
    visually point at its own column. Call after the axes limits are final;
    a later relayout (tight_layout etc.) invalidates the correction.
    """
    if ticks is None:
        ticks = list(ax.get_xticks())
    if fontsize is None:
        fontsize = PUBLICATION_FONTS['tick_label']

    ax.set_xticks(list(ticks))
    ax.set_xticklabels(labels, rotation=rotation, ha='center', va='top',
                       rotation_mode='anchor', fontsize=fontsize,
                       fontfamily='Times New Roman')
    ax.tick_params(axis='x', pad=pad)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for tick, label in zip(ticks, ax.get_xticklabels()):
        bb = label.get_window_extent(renderer=renderer)
        tick_px = ax.transData.transform((float(tick), 0.0))[0]
        shift_px = tick_px - 0.5 * (bb.x0 + bb.x1)
        x0_px = ax.transData.transform((float(tick) - 1.0, 0.0))[0]
        px_per_data = tick_px - x0_px
        shift_data = shift_px / px_per_data if px_per_data else 0.0
        x_pos, y_pos = label.get_position()
        label.set_position((x_pos + shift_data, y_pos))


def add_statistical_annotations(
    ax,
    x_positions,
    statistics,
    y_offset: float = 5,
    fontsize: int | None = None,
    color: str = '#2C3E50',
):
    """Annotate bars with mean±std labels following PRISM conventions."""
    if fontsize is None:
        fontsize = PUBLICATION_FONTS['annotation']

    for x_pos, stats in zip(x_positions, statistics.values()):
        if 'mean' not in stats or 'std' not in stats:
            continue

        mean_val = stats['mean']
        std_val = stats['std']

        if 'values' in stats:
            max_height = max(stats['values'])
            y_pos = max_height + std_val + y_offset
        else:
            y_pos = mean_val + std_val + y_offset

        ax.text(
            x_pos,
            y_pos,
            f"{mean_val:.1f}±{std_val:.1f}",
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color,
            weight='bold',
            bbox={
                'boxstyle': 'round,pad=0.3',
                'facecolor': 'white',
                'alpha': 0.9,
                'edgecolor': 'gray',
                'linewidth': 0.8,
            },
        )


def style_axes_for_publication(
    ax,
    spine_width: float = 1.5,
    tick_width: float = 1.5,
    tick_length: float = 6,
    grid_alpha: float = 0.6,
):
    """Apply PRISM axis styling to arbitrary matplotlib axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)

    ax.tick_params(axis='both', which='major', width=tick_width, length=tick_length)
    ax.tick_params(
        axis='both', which='minor', width=tick_width * 0.7, length=tick_length * 0.5
    )

    ax.grid(True, alpha=grid_alpha, linewidth=1.0, linestyle='-')
    ax.set_axisbelow(True)


def format_colorbar(cbar, label: str = '', fontsize: int | None = None):
    """Apply publication-quality styling to a colorbar.

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object to format
    label : str
        Label for the colorbar
    fontsize : int, optional
        Font size for colorbar labels and ticks. If None, uses PUBLICATION_FONTS['colorbar']

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(data, cmap='viridis')
    >>> cbar = fig.colorbar(im, ax=ax)
    >>> format_colorbar(cbar, label='Value (a.u.)')

    """
    if fontsize is None:
        fontsize = PUBLICATION_FONTS['colorbar']

    # Set tick label size
    cbar.ax.tick_params(labelsize=fontsize)

    # Set colorbar label
    if label:
        cbar.set_label(label, fontsize=fontsize, fontweight='bold',
                      family='Times New Roman')

    # Ensure colorbar axis uses Times New Roman for all text
    for text in cbar.ax.get_yticklabels():
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(fontsize)


def format_axis_labels(ax, xlabel=None, ylabel=None, title=None):
    """Format axis labels with consistent styling."""
    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=PUBLICATION_FONTS['axis_label'],
            fontweight="bold",
            fontfamily="Times New Roman",
        )
    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=PUBLICATION_FONTS['axis_label'],
            fontweight="bold",
            fontfamily="Times New Roman",
        )
    if title:
        ax.set_title(
            title,
            fontsize=PUBLICATION_FONTS['title'],
            fontweight="bold",
            fontfamily="Times New Roman",
            pad=20,
        )


def add_annotation(ax, text, xy, xytext=None, fontsize=None):
    """Add annotation with consistent styling."""
    if fontsize is None:
        fontsize = PUBLICATION_FONTS['annotation']

    if xytext is None:
        xytext = (xy[0] + 0.1, xy[1] + 0.1)

    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        ha="left",
        va="bottom",
        fontfamily="Times New Roman",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
    )


def create_performance_metric_labels(metric_type="pearson_r"):
    """Create proper metric labels for performance charts."""
    metric_labels = {
        "pearson_r": "Pearson Correlation Coefficient (r)",
        "r2": "R^2 Score",
        "rmse": "Root Mean Square Error (RMSE)",
        "mae": "Mean Absolute Error (MAE)",
    }
    return metric_labels.get(metric_type, metric_type)


# ---------------------------------------------------------------------------
# Saving and subplot stitching
# ---------------------------------------------------------------------------

def save_publication_figure(
    fig,
    save_path,
    dpi: int = 300,
    format: str = 'png',
    bbox_inches: str = 'tight',
    transparent: bool = False,
):
    """Persist a figure with consistent DPI/bounds for publications."""
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=dpi,
        format=format,
        bbox_inches=bbox_inches,
        facecolor='white' if not transparent else None,
        edgecolor='none',
        transparent=transparent,
    )
    logger.info("Publication figure saved: %s (%s, %d DPI)", save_path, format, dpi)


def save_figure(fig, filename, destination=None, formats=None, dpi=300):
    """Save figure into the appropriate destination directory."""
    dest_key = destination or FIGURE_DESTINATIONS.get(filename, DEFAULT_DESTINATION)
    dest_dir = DESTINATION_PATHS.get(dest_key, REPORT_FIGURE_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["png"]

    base_path = dest_dir / filename

    for fmt in formats:
        save_path = str(base_path) + f".{fmt}"
        save_publication_figure(fig, save_path, dpi=dpi, format=fmt)

    plt.close(fig)


def create_publication_colormap(name: str = 'scientific', n_colors: int = 256):
    """Return a listed colormap consistent with PRISM palettes."""
    from matplotlib.colors import ListedColormap

    if name == 'scientific':
        colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
    elif name == 'heatmap':
        colors = ['#ffffcc', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
    elif name == 'diverging':
        colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
    else:
        colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']

    return ListedColormap(colors, name=name, N=n_colors)


def get_anion_colors() -> dict[str, str]:
    """Expose consistent colors for the five anion groups."""
    return ANION_COLORS


def combine_subplots_horizontal(
    subplot_files, output_file, gap_pixels: int = 200, label_font: str = 'Times New Roman'
):
    """Combine subplot PNGs horizontally while adding PRISM-style labels."""
    import os

    import matplotlib.image as mpimg

    images = []
    for f in subplot_files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping...")
            continue
        images.append(mpimg.imread(f))

    if not images:
        print("Error: No images found!")
        return

    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_height = max(heights)
    total_width = sum(widths) + gap_pixels * (len(images) - 1)

    combined = np.ones((max_height, total_width, 4), dtype=np.float32)

    x_offset = 0
    for img in images:
        h, w = img.shape[0], img.shape[1]
        y_offset = (max_height - h) // 2
        combined[y_offset:y_offset + h, x_offset:x_offset + w, :] = img
        x_offset += w + gap_pixels

    dpi = 300
    fig_width = total_width / dpi
    fig_height = max_height / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(combined)
    ax.axis('off')

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    x_positions = [0]
    for w in widths[:-1]:
        x_positions.append(x_positions[-1] + w + gap_pixels)

    for label, x_pos in zip(labels[:len(images)], x_positions):
        label_x = (x_pos - 30) / total_width
        label_y = 1.02
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=PUBLICATION_FONTS['title'],
            fontweight='bold',
            family=label_font,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.1,
    )
    plt.close()

    print(f"✓ Combined figure saved: {output_file}")
    print(f"  Total size: {total_width} x {max_height} pixels")


# Apply publication style globally so every Matplotlib object (legends,
# annotations, colorbars, etc.) inherits the same font family/size without
# requiring per-call overrides.
apply_publication_style('default')
