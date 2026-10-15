#!/usr/bin/env python3
"""Manuscript Figure 4: multimodal model-screening landscape.

Answers the Results question "is any single representation or algorithm
consistently optimal across chemical classes?" from the complete screening
output:

    (A) distribution of the three-fold CV mean r (per valid initial-screening
        record) over ALL screened representation-model configurations, per
        target class, strict vs few-shot LOGO side by side;
    (B) best three-fold CV mean r per representation family;
    (C) best three-fold CV mean r per model family.

Data source:
    the public Supplementary Data 1 export
    manuscript/Supplementary_Data/Data1_LOGO_screening_output.csv, regenerated
    from the internal source ``data/logo_screening_output.csv`` by
    scripts/figures/manuscript/export_data1_screening_output.py (the internal
    source keeps the historical column names and bookkeeping columns;
    this figure reads the public reader-facing schema).
    Valid initial-screening records with a three-fold CV: 3,507 strict +
    3,501 few-shot = 7,008 fold-level records. 834 of the 7,842 stage-1 rows
    are excluded (746 all-zero failed placeholders, 55 with fewer than three
    folds, 33 with non-finite values), and all stage-2 HPO rows are excluded
    because their CV scores merely inherit the stage-1 values and would be
    drawn as duplicate configurations.

Known asymmetry (2026-10-02): the canonical Data 1 export originally contained
76 deepchem_mol2vec rows that existed ONLY for few-shot LOGO / target class
A/N (the 2026-09-05 rerun_improved_AN re-screening with the complete
representation set). No other class or setting ever evaluated mol2vec, so its
score could raise the A/N few-shot cell of the "2D fingerprints"
representation family (panel b) and whichever model family scored highest on
it (panel c) without a fair counterpart. Those 76 rows have been removed from
Supplementary Data 1 itself, so --exclude-postlock-mol2vec is now a no-op
safeguard kept for older copies of the CSV that still contain them.

The plotted three-fold CV mean r is computed per record from the
initial_screening_cv_pearson_r_by_fold column (three folds on the
model-development subset of the LOGO training set). The separate
internal_validation_pearson_r column (Pearson r on the internal 20% check
subset of that partition, the ranking criterion used for model selection) is
intentionally NOT shown: the figure reports the cross-validation diagnostics,
not the selection criterion.

Notation (user decision 2026-09-02): ML/LOGO figures use the five merged
classes A/N, C, H, P, S (the slash pairs the hA/OH- and hN/O- subgroups and
is NOT a subscript); no chemical formulas inside the figure.

Output: manuscript/Figures/Fig4_Screening_Landscape.png
(shared common_figures and patent figures are not touched).

Usage:
    python plot_logo_landscape.py [--exclude-postlock-mol2vec]
                                  [--output PATH]
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from plot_utils import (
    FEWSHOT_COLOR,
    FEWSHOT_FILL,
    PUBLICATION_FONTS,
    STRICT_COLOR,
    STRICT_FILL,
    apply_publication_style,
    lighten_color,
    save_publication_figure,
)
from public_data import load_logo_screening

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BASE_DIR / 'scripts'
OUTPUT_PATH = (
    BASE_DIR / 'manuscript' / 'Figures' / 'Fig4_Screening_Landscape.png'
)

CLASS_ORDER = ['A_N', 'C', 'H', 'P', 'S']
CLASS_LABELS = {'A_N': 'A/N', 'C': 'C', 'H': 'H', 'P': 'P', 'S': 'S'}
SETTINGS = [('strict', 'Strict'), ('improved', 'Few-shot')]

# Representation family mapping (mirrors the Supplementary Table S2
# categories; EState is a descriptor set, ECIF/NNScore/E3FP are 3D-derived).
# Short display names keep the heatmap row labels from invading panel A.
REP_FAMILY = {
    '2D fingerprints': [
        'morgan_fp_r2_512', 'morgan_fp_r2_1024', 'morgan_fp_r2_2048',
        'morgan_fp_r3_1024', 'morgan_fp_r3_2048', 'morgan_feature_fp_r2',
        'morgan_feature_fp_r3', 'rdkit_fp_512', 'rdkit_fp_1024',
        'rdkit_fp_2048', 'maccs_keys', 'atom_pair_fp', 'torsion_fp',
        'datamol_ecfp4', 'datamol_ecfp6', 'datamol_ecfp4_1024',
        'datamol_ecfp4_count', 'datamol_fcfp4', 'datamol_fcfp6',
        'datamol_fcfp4_count', 'datamol_topological', 'datamol_atompair',
        'datamol_atompair_count', 'datamol_avalon', 'datamol_erg',
        'datamol_maccs', 'datamol_rdkit', 'datamol_secfp',
        'deepchem_morgan_r2_2048', 'deepchem_morgan_r3_1024',
        'deepchem_maccs', 'deepchem_mol2vec', 'cdk',
    ],
    'Descriptors': ['rdkit_all_descriptors', 'datamol_estate'],
    '3D descriptors': ['nnscore_3d', 'ecif_3d', 'e3fp_3d'],
    'Matrix': ['coulomb_matrix', 'adjacency_matrix'],
    'Image': ['deepscreen_image', 'chemception_image', 'rdkit_2d_image',
              'imagemol_style'],
    'String': ['canonical_smiles', 'inchi', 'inchikey'],
}

# Model family mapping over the 22 screened algorithms.
MODEL_FAMILY = {
    'Linear': ['ridge', 'lasso', 'elastic_net', 'bayesian_ridge',
               'svm_linear'],
    'RBF-SVM': ['svm_rbf'],
    'Tree ensembles': ['random_forest', 'extra_trees', 'gradient_boosting',
                       'xgboost', 'lightgbm', 'bagging', 'decision_tree'],
    'kNN': ['knn'],
    'MLP': ['mlp', 'mlp_transformer_fallback'],
    'CNN': ['matrix_cnn', 'matrix_cnn_small', 'image_cnn', 'image_cnn_small'],
    'Transformer': ['transformer_small', 'transformer_medium'],
}


def parse_cv_scores(value: object) -> Optional[list[float]]:
    """Parse an ``initial_screening_cv_pearson_r_by_fold`` cell into a float list, or None."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        return [float(t) for t in str(value).strip('[]').split(',') if t.strip()]
    except ValueError:
        return None


def valid_cv_mean(value: object) -> Optional[float]:
    """Three-fold CV mean of one record, or None when the record is invalid.

    A record counts only when it carries at least three finite per-fold
    values that are not the all-zero placeholder written for failed runs.
    """
    scores = parse_cv_scores(value)
    if not scores or len(scores) < 3:
        return None
    if not all(np.isfinite(s) for s in scores):
        return None
    if all(abs(s) < 1e-9 for s in scores):
        return None
    return float(np.mean(scores))


def load_screening_data() -> pd.DataFrame:
    """Load the public Supplementary Data 1 screening table.

    Thin wrapper over ``public_data.load_logo_screening``: one schema-aware
    implementation for all public readers, so the fallback/schema-gate logic
    cannot drift between this figure and the other loaders.
    """
    return load_logo_screening()


def family_lookup(family_map: dict):
    """Return a name -> family-key function for the given mapping."""
    def lookup(name: str) -> str:
        for family, members in family_map.items():
            if name in members:
                return family
        raise KeyError(f"Unmapped screening entry: {name}")
    return lookup


def draw_violin_panel(ax, scores: dict) -> None:
    """Panel A: CV-score distributions per class, strict vs few-shot."""
    positions = np.arange(len(CLASS_ORDER))
    offsets = {'strict': -0.20, 'improved': +0.20}
    # Match the Figure 5 bar fills exactly: Fig. 5 uses the same
    # STRICT_FILL/FEWSHOT_FILL facecolours but draws its bars at alpha=0.85
    # over the white canvas, so its rendered colour is FILL composited onto
    # white at 0.85 (#a0c5e0 / #ffc898). Baking that same composite into the
    # violin facecolour (and forcing body alpha to 1.0 below, since
    # matplotlib's violinplot leaves bodies at alpha=0.3 and would otherwise
    # double-fade the pre-blended fill) gives pixel-identical fills in both
    # figures without touching plot_utils or plot_logo_performance.
    colors = {'strict': lighten_color(STRICT_FILL, 0.85),
              'improved': lighten_color(FEWSHOT_FILL, 0.85)}

    for setting, _ in SETTINGS:
        data = [scores[(setting, c)] for c in CLASS_ORDER]
        parts = ax.violinplot(
            data,
            positions=positions + offsets[setting],
            widths=0.36,
            showmeans=True,
            showextrema=False,
            showmedians=False,
        )
        for body in parts['bodies']:
            # Soft fill matches the Figure 5 bars (same FILL_ALPHA blend) and
            # stays lighter than the Fig. S5/S6 scatter points.
            body.set_facecolor(colors[setting])
            # matplotlib's violinplot leaves bodies at alpha=0.3; with the
            # pre-blended facecolour that would double-fade it to 15%
            # effective opacity (the original "violin looks lighter than the
            # Fig. 5 bars" bug), so force full opacity.
            body.set_alpha(1.0)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.4)

    ax.axhline(0.0, color='gray', linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                       fontsize=PUBLICATION_FONTS['tick_label'])
    ax.set_xlim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.set_ylabel('Screening CV mean $r$',
                  fontsize=PUBLICATION_FONTS['axis_label'], fontweight='bold')
    ax.set_xlabel('Target class', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontweight='bold')
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    # Explicit linewidth: the shared style sets grid.linewidth=0, which turns
    # dashed grid lines into an all-zero dash pattern and breaks drawing.
    ax.grid(False)
    ax.set_axisbelow(True)


def _wrap_row_label(label: str) -> str:
    """Wrap the long model-family name so panel C's row labels stay
    inside the gap between the two second-row heatmaps."""
    return {'Tree ensembles': 'Tree\nensembles'}.get(label, label)


def _fig2b_soft_red_cmap() -> mcolors.LinearSegmentedColormap:
    """Return the RdBu_r variant whose red end matches Figure 2b.

    Figure 2b (plot_data_analysis.create_deltaG_heatmap_six_subgroup_formulas)
    maps its RdBu_r through seaborn with symmetric centring on the dataset
    mean: seaborn extends vmin/vmax symmetrically around the centre to
    ``center +/- vrange`` (vrange = the larger of the two deviations) and
    truncates the colormap to the part the data actually occupy. The reddest
    displayed cell (the C-row maximum, 4.25) therefore lands at fraction
    (pivot_max - (center - vrange)) / (2 * vrange) of RdBu_r -- a soft brick
    red (#dc6e57), far lighter than the full endpoint (#67001f). This helper
    recomputes that fraction from the public reference table so Fig. 4's r = 1.0 cell shows
    exactly the Fig. 2b red. The blue (low) end starts at RdBu_r(0.2)
    (#4393c3, a medium blue) instead of the full navy (#053061): this avoids an
    over-saturated deep blue, and is harmless because the panels B/C data
    minima are >= 0.51, so nothing below the 0.4 colour-scale lower bound is
    ever displayed (see HEATMAP_VMIN).
    """
    from plot_data_analysis import classify_r_group, load_molecular_data

    df = load_molecular_data()
    data = df.copy()
    data['r_group'] = data['Name'].apply(classify_r_group)
    data['subgroup'] = data['Name'].str[1].str.upper()
    pivot = data.pivot_table(values='dG_rel_kcal_per_mol', index='subgroup',
                             columns='r_group', aggfunc='mean')

    center = df['dG_rel_kcal_per_mol'].mean()
    vrange = max(pivot.max().max() - center, center - pivot.min().min())
    red_fraction = (pivot.max().max() - (center - vrange)) / (2 * vrange)

    base = plt.get_cmap('RdBu_r')
    # Blue (low) end starts at 0.2 (#4393c3) rather than 0.0 (#053061 navy) so
    # the deepest blue is not over-saturated (request 2026-09-12). White sits at
    # the natural RdBu_r midpoint (RdBu_r = 0.5) of this slice, i.e. ramp fraction
    # (0.5 - BLUE_START) / (red_fraction - BLUE_START); with vmin = 0.4 that puts
    # the white point at r ~= 0.73, keeping the blue->white and white->red bands
    # balanced (request 2026-09-12: an earlier pull-down of white made the cells
    # less distinct, so white is left centred here).
    BLUE_START = 0.2
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'RdBu_r_fig2b', base(np.linspace(BLUE_START, red_fraction, 256)))
    cmap.set_bad('#d9d9d9')
    return cmap


# Cell-value range of panels B/C (CV Pearson r).
# Originally HEATMAP_VMIN = -0.1. Raised to 0.4 on 2026-09-12: the best-r
# minima across the B/C cells are >= 0.51, so the lower end sits just below the
# data and the 0.5-0.9 band spreads across most of the blue->red ramp,
# separating the 0.7/0.8/0.9 cells. (Change HEATMAP_VMIN back to -0.1 to
# reproduce the prior figure.)
HEATMAP_VMIN = 0.4
HEATMAP_VMAX = 1.0


def draw_family_heatmap(ax, best: dict, row_order: list):
    """Panel B/C: best CV score per family, 5 classes x 2 settings.

    Column blocks: left half strict, right half few-shot. The setting is
    encoded by the tick-label colour (blue = strict, orange = few-shot, as
    in the figure legend), which avoids repeating long two-line labels over
    the narrow columns. The colormap is the same diverging RdBu_r family as
    the PMF heatmap of Figure 2b, truncated so the saturation of the red
    end matches Figure 2b exactly (see _fig2b_soft_red_cmap).
    """
    n_cls = len(CLASS_ORDER)
    matrix = np.full((len(row_order), 2 * n_cls), np.nan)
    for col, cls in enumerate(CLASS_ORDER):
        matrix[:, col] = [best[('strict', cls)].get(f, np.nan)
                          for f in row_order]
        matrix[:, n_cls + col] = [best[('improved', cls)].get(f, np.nan)
                                  for f in row_order]

    cmap = _fig2b_soft_red_cmap()
    norm = mcolors.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    def text_color(value: float) -> str:
        """White on the dark ends of the map, black on light cells.

        Threshold on the relative luminance of the actual mapped colour
        (white when luminance < 0.45) instead of a hard value cut. With the
        current lighter-blue / soft-red ramp both ends sit above the
        threshold, so all cell text is black; the check is kept so that if
        the ramp is darkened again the low (blue) end automatically switches
        to white text.
        """
        red, green, blue, _ = cmap(norm(value))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return 'white' if luminance < 0.45 else 'black'

    # Per-column maximum: bold the best-scoring family in each class/setting
    # column so the strongest representation/model family reads at a glance.
    col_max = np.nanmax(matrix, axis=0)

    # Annotate every cell; families absent from a fold's run get a dash.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f'{value:.2f}' if np.isfinite(value) else '–'
            dark_cell = np.isfinite(value) and text_color(value) == 'white'
            is_col_max = np.isfinite(value) and value == col_max[j]
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=PUBLICATION_FONTS['tick_label'] - 8,
                    fontfamily='Times New Roman',
                    fontweight='bold' if is_col_max else 'normal',
                    color='white' if dark_cell else 'black')

    # Divider between the strict (left) and few-shot (right) column blocks.
    ax.axvline(n_cls - 0.5, color='black', linewidth=1.6)

    ax.set_xticks(np.arange(2 * n_cls))
    ax.set_xticklabels([CLASS_LABELS[c] for _ in SETTINGS for c in CLASS_ORDER],
                       fontsize=PUBLICATION_FONTS['tick_label'] - 3)
    # Class tick labels stay black: the setting (strict/few-shot) is now
    # distinguished by the explicit block labels below, so colouring the class
    # labels would only add visual noise.
    for tick_label in ax.get_xticklabels():
        tick_label.set_color('black')
        tick_label.set_fontweight('bold')
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([_wrap_row_label(f) for f in row_order],
                       fontsize=PUBLICATION_FONTS['tick_label'] - 4)
    # Heatmaps keep the shared 2.0 spine frame but no tick marks, which would
    # collide with the cell text.
    ax.tick_params(axis='both', which='both', length=0)

    # Explicit "Strict LOGO" / "Few-shot LOGO" block labels directly under the
    # class tick labels. The two column blocks are otherwise distinguished only
    # by position, which relies on the panel-A legend; spelling the setting out
    # makes panels B/C self-contained (each figure readable on its own, per the
    # manuscript figure rules) and is cheaper than a bracket. Class labels are
    # black, so the blue/orange block labels carry the setting encoding.
    # transAxes y below the x-axis keeps the text glued regardless of data-row
    # count; lowered from -0.055 to -0.095 (request 2026-09-10) so it reads as
    # an x-axis label rather than a tick annotation. Kept bold.
    for x_mid, label, color in [
            (0.25, 'Strict LOGO', STRICT_COLOR),
            (0.75, 'Few-shot LOGO', FEWSHOT_COLOR),
    ]:
        ax.text(x_mid, -0.095, label, ha='center', va='top',
                transform=ax.transAxes,
                fontsize=PUBLICATION_FONTS['tick_label'] + 2,
                fontfamily='Times New Roman', fontweight='bold',
                color=color)
    return im


def main() -> None:
    """Assemble the screening-landscape figure and save it."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--exclude-postlock-mol2vec', action='store_true',
        help='Drop the 76 deepchem_mol2vec rows that exist only for '
             'few-shot LOGO / target class A/N (post-lock re-screening), '
             'so the family heatmaps are not raised by a representation '
             'that no other class or setting ever evaluated.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Alternative output PNG path (default: the manuscript '
             'Fig4_Screening_Landscape.png). Use a different path to '
             'compare variants without overwriting the manuscript figure.',
    )
    args = parser.parse_args()

    rep_of = family_lookup(REP_FAMILY)
    model_of = family_lookup(MODEL_FAMILY)

    # Load screening data from CSV (single source of truth for consistency)
    df_all = load_screening_data()
    # The published Supplementary Data 1 labels this target class "A/N";
    # historical exports used "A_N". Normalise to the internal key so the
    # figure builds identically from either source.
    df_all['held_out_class'] = df_all['held_out_class'].replace({'A/N': 'A_N'})

    # Figure 4 reports the three-fold cross-validation mean of the
    # model-development subset, computed per record from the per-fold diagnostic
    # column. Stage-2 (HPO) rows inherit the identical CV scores from their
    # stage-1 counterpart and would be drawn as duplicate configurations, so
    # only valid initial-screening records are kept (exclusion counts in the
    # module docstring).
    df_all = df_all[df_all['screening_stage'] == 1].copy()
    df_all['cv_mean'] = df_all['initial_screening_cv_pearson_r_by_fold'].map(valid_cv_mean)
    df_all = df_all[df_all['cv_mean'].notna()]

    if args.exclude_postlock_mol2vec:
        mask = (
            (df_all['setting'] == 'Few-shot LOGO')
            & (df_all['held_out_class'] == 'A_N')
            & (df_all['representation'] == 'deepchem_mol2vec')
        )
        n_excluded = int(mask.sum())
        df_all = df_all[~mask].copy()
        print(f"Excluded {n_excluded} post-lock A/N few-shot mol2vec rows "
              f"(remaining records: {len(df_all)})")

    scores = {}
    best = {}
    total = {'strict': 0, 'improved': 0}

    for setting, _ in SETTINGS:
        for cls in CLASS_ORDER:
            setting_col = 'Strict LOGO' if setting == 'strict' else 'Few-shot LOGO'
            df = df_all[(df_all['setting'] == setting_col) &
                       (df_all['held_out_class'] == cls)].copy()

            if df.empty:
                print(f"  Warning: no data for {setting_col}/{cls}")
                continue

            total[setting] += len(df)
            scores[(setting, cls)] = df['cv_mean'].to_numpy()
            df = df.assign(
                rep_family=df['representation'].map(rep_of),
                model_family=df['model'].map(model_of),
            )
            best[(setting, cls)] = {
                **df.groupby('rep_family')['cv_mean'].max().to_dict(),
                **df.groupby('model_family')['cv_mean'].max().to_dict(),
            }

    print(f"Valid initial-screening CV records: strict {total['strict']}, "
          f"few-shot {total['improved']}, total "
          f"{total['strict'] + total['improved']}")

    # Shared manuscript style: axes.linewidth 2.0, ticks out, Times New Roman
    # (the previous local override with 1.2/inward ticks mismatched Fig. 2/3/5).
    apply_publication_style()

    # Rows omitted from the family heatmaps (2026-09-10): string encodings and
    # transformer models were screened and still count towards the panel (a)
    # distributions and remain in Supplementary Data 1, but they are dropped
    # from panels (b)/(c) to keep the heatmaps readable. The family maps above
    # stay complete so every screening entry still resolves to a family.
    omitted_rows = {'String', 'Transformer'}
    rep_rows = [f for f in REP_FAMILY if f not in omitted_rows]
    model_rows = [f for f in MODEL_FAMILY if f not in omitted_rows]

    # Two-row layout: the violin panel spans the full first row (a 1x3 layout
    # squeezed both the violins and the heatmap cells), and the two family
    # heatmaps share the second row so their cells stay close to square.
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.0],
                          hspace=0.38, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_violin_panel(ax_a, scores)

    ax_b.set_title('Representation family',
                   fontsize=PUBLICATION_FONTS['subtitle'],
                   fontweight='bold', pad=12)
    im_b = draw_family_heatmap(ax_b, best, rep_rows)
    ax_c.set_title('Model family',
                   fontsize=PUBLICATION_FONTS['subtitle'],
                   fontweight='bold', pad=12)
    draw_family_heatmap(ax_c, best, model_rows)

    # Single-axes colorbar keeps the figure tight_layout-compatible (a
    # two-axes colorbar made save_publication_figure's tight_layout no-op and
    # previously let the heatmap row labels collide with neighbouring panels).
    cbar = fig.colorbar(im_b, ax=ax_c, fraction=0.05, pad=0.02)
    cbar.set_label('Best screening CV mean $r$',
                   fontsize=PUBLICATION_FONTS['colorbar'], fontweight='bold')
    cbar.ax.tick_params(labelsize=PUBLICATION_FONTS['colorbar'])

    # The strict/few-shot palette is introduced by panel A, so its legend
    # sits above that panel instead of below the whole figure.
    handles = [
        mpatches.Patch(facecolor=STRICT_FILL, edgecolor='black',
                       linewidth=0.8, label='Strict LOGO'),
        mpatches.Patch(facecolor=FEWSHOT_FILL, edgecolor='black',
                       linewidth=0.8, label='Few-shot LOGO'),
    ]
    ax_a.legend(handles=handles, loc='lower right',
                bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False,
                prop={'family': 'Times New Roman',
                      'size': PUBLICATION_FONTS['legend']})

    # Panel letters: a/b sit just left of their own left spine (above the
    # panel), aligned by an absolute inch offset so they clear the y-axis
    # labels. Panel c is centred in the b-c gap (x = -0.11 in c's axes
    # fraction, i.e. the gap midpoint, centre-aligned) so it stays clear of
    # panel b yet is not shoved inside panel c (per request: first moved too
    # far into c, now pulled back to the gap midpoint).
    label_x = {
        'a': -2.2 / (ax_a.get_position().width * fig.get_figwidth()),
        'b': -2.2 / (ax_b.get_position().width * fig.get_figwidth()),
        'c': -0.11,
    }
    label_ha = {'a': 'right', 'b': 'right', 'c': 'center'}
    for letter, ax in [('a', ax_a), ('b', ax_b), ('c', ax_c)]:
        ax.text(label_x[letter], 1.02, letter, transform=ax.transAxes,
                fontsize=42, fontweight='bold', fontfamily='Times New Roman',
                verticalalignment='bottom', horizontalalignment=label_ha[letter])

    output_path = (Path(args.output) if args.output else OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, str(output_path))
    print(f"  saved: {output_path}")


if __name__ == '__main__':
    main()
