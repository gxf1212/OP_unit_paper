#!/usr/bin/env python3
"""Manuscript Figure 5: cross-class transferability under strict and few-shot LOGO.

Four paired panels over the five LOGO classes (A/N, C, H, P, S) in a 2x2 grid:
    a. three-fold CV mean r of the selected representation-model pair,
       computed on the complete LOGO training set with the initial-screening
       parameters fixed (standardization
       fitted within each training fold; the separate screening check-set
       selection score, primary_metric, is not shown)
    b. external test Pearson r
    c. external test MAE (kcal mol^-1)
    d. external test mean signed error (pred - true, kcal mol^-1): bars are
       fold means, jittered points are the per-sample signed errors, and a
       solid zero line exposes class-specific over-/under-prediction

Data sources are resolved at runtime: each source prefers the release
``data/`` CSV copy, falling back to the working-repository location. The
release build ships only the ``data/`` copies (the internal ``scripts/``
locations and the retrained-model ``predictions.json`` files are not
shipped), so the same script serves both the released checkout and the
working repository without path rewriting:
    data/logo_cv_full_outer_summary.csv (release) / the audited CV summary
        in the working repository -- three-fold CV mean r on
        the complete LOGO training set with fixed initial-screening
        parameters (panel a)
    data/logo_metrics.csv (release) / the external metrics summary in the
        working repository -- external test Pearson r and MAE of the
        retrained final models (panels b/c)
    data/logo_external_predictions.csv (release), derived by the release
        build from the retrained-model predictions.json files; the working
        repository falls back to reading results_logo_{strict,improved}/fold_*/
        retrained/predictions.json directly (panel d)
Rows with logo_type='strict' are Strict LOGO; logo_type='improved' is the
few-shot LOGO setting (one/two lead motifs added to training). The two
summaries are merged one-to-one on (setting, target class, representation,
model). The separate screening selection score (primary_metric, recorded on
the internal 20% check subset and used to rank configurations) is
deliberately not shown; panel a reports CV on the complete LOGO training set,
not the screening CV and not the CV of the tuned final model.

Panel d is computed per-sample from the retrained final models (mean of
test.y_pred - test.y_true for each fold). The strict model shows a
near-unilateral positive bias for A/N (+2.13 kcal mol^-1, all 16 samples over-
predicted) and a negative bias for C/H, i.e. regression toward the training-set
mean for a fully held-out class; few-shot augmentation shrinks the magnitudes
without reversing their signs.

Terminology/notation (user decision 2026-09-02, rev.): all manuscript
figures use single-letter class codes. ML/LOGO figures merge the hA/OH- and
hN/O- terminal subgroups into one class written "A/N" (a slash pairing, NOT
a subscript); descriptive figures show the six subgroups A, N, C, H, P, S.
No chemical-formula tick labels anywhere — captions decode the codes.

Output: manuscript/Figures/Fig5_CrossClass_Transferability.png
(shared common_figures and patent figures are not touched; renamed from the
initial Fig4_LOGO_Strict_vs_FewShot.png on 2026-09-03 when Figure 4 was
reassigned to the screening-landscape figure per plan_v0.md).

Usage:
    python plot_logo_performance.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from plot_utils import (
    FEWSHOT_COLOR,
    FEWSHOT_FILL,
    PUBLICATION_FONTS,
    STRICT_COLOR,
    STRICT_FILL,
    save_publication_figure,
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / 'manuscript' / 'Figures' / 'Fig5_CrossClass_Transferability.png'
RESULTS_ROOT = BASE_DIR / 'scripts'


def _resolve_data(*paths: Path) -> Path:
    """Return the first existing path, release ``data/`` copy preferred.

    The release build ships only the ``data/`` CSV copies; the working
    repository keeps the same information in internal ``scripts/`` locations.
    Resolving at import time lets one source script serve both layouts; the
    release sed rules only rewrite contiguous docstring strings, so the code
    paths must resolve themselves rather than rely on patching.
    """
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "no data source found among:\n  " + "\n  ".join(str(p) for p in paths))


SUMMARY_CSV = _resolve_data(
    BASE_DIR / 'data' / 'logo_metrics.csv',
    BASE_DIR / 'scripts' / 'unknown' / 'appendix' / 'logo_metrics.csv',
)
CORRECTED_CV_CSV = _resolve_data(
    BASE_DIR / 'data' / 'logo_cv_full_outer_summary.csv',
    (BASE_DIR / 'scripts' / 'analysis' / 'logo_cv_corrected'
     / 'selected_initial_full_outer_cv_summary.csv'),
)
EXTERNAL_PRED_CSV = BASE_DIR / 'data' / 'logo_external_predictions.csv'

# Five LOGO classes in fold order, drawn with the ML/LOGO single-letter
# codes: the merged amine-oxide-related class is written "A/N" (a slash
# pairing of the hA/OH- and hN/O- terminal subgroups, NOT a subscript),
# the other four keep their dataset codes C/H/P/S. The caption decodes the
# codes; descriptive figures (2a/2b/3, SI S2/S4) use the six single letters.
CLASS_ORDER = ['A_N', 'C', 'H', 'P', 'S']
CLASS_LABELS = {
    'A_N': 'A/N',
    'C': 'C',
    'H': 'H',
    'P': 'P',
    'S': 'S',
}

# (column, ylabel, ylim) per panel. Bars start at zero, so all ylims do too.
# Pearson/MAE wording is kept short per the figure conventions; the caption
# defines r as the Pearson coefficient. Panel a draws the
# LOGO training-set CV (cv_mean_r from the audited summary), not the historical
# screening-time cv_mean column.
PANEL_SPECS = [
    ('cv_mean_r', 'Training-set CV $r$', (0.0, 1.05)),
    ('test_pearson_r', 'External-test $r$', (0.0, 1.05)),
    ('test_mae', 'External-test MAE\n' + r'($\mathrm{kcal\cdot mol^{-1}}$)', (0.0, 2.40)),
]

# Panel d is a signed quantity spanning both sides of zero, so its ylim is
# symmetric around the drawn zero line instead of starting at zero. The
# range must also contain the extreme per-sample residuals (strict A/N
# +3.18, strict C -2.51), not just the fold means. The mean signed error is
# labeled MBE (mean bias error, its standard abbreviation) to parallel the
# MAE ylabel of panel c; the caption decodes the abbreviation.
SIGNED_ERR_YLABEL = 'External-test MBE\n' + r'($\mathrm{kcal\cdot mol^{-1}}$)'
SIGNED_ERR_YLIM = (-3.5, 3.5)


def load_logo_summary() -> pd.DataFrame:
    """Merge the audited LOGO training-set CV with external-test metrics (1:1).

    Panel a draws the three-fold CV mean *r* computed on the complete LOGO
    training set with the fixed initial-screening model parameters
    (``selected_initial_full_outer_cv_summary.csv``). Panels
    b/c draw the external-test *r* and MAE of the retrained final models
    (``logo_metrics.csv``). The two summaries are merged one-to-one
    on (setting, target class, representation, model) so each CV value is
    paired with the external-test result of the very same representation-
    model pair; a missing, duplicated or mismatched key fails the build.
    """
    strategy_map = {'strict': 'Strict LOGO', 'improved': 'Few-shot LOGO'}
    corrected = pd.read_csv(CORRECTED_CV_CSV)
    corrected = corrected.loc[corrected['configuration'] == 'initial']
    external = pd.read_csv(SUMMARY_CSV)
    merged = corrected.merge(
        external[['logo_type', 'fold', 'representation', 'model',
                  'test_pearson_r', 'test_mae']],
        left_on=['setting', 'held_out_class', 'representation', 'model'],
        right_on=['logo_type', 'fold', 'representation', 'model'],
        how='inner',
        validate='one_to_one',
    )
    expected = 2 * len(CLASS_ORDER)  # ten selected representation-model pairs
    if len(merged) != expected:
        raise ValueError(
            f"LOGO training-set CV / external summary merge produced {len(merged)} rows, "
            f"expected {expected} (a representation/model mismatch is likely)"
        )
    if merged.duplicated(['setting', 'held_out_class']).any():
        raise ValueError("duplicate (setting, target class) keys after merge")
    df = pd.DataFrame({
        'fold': merged['held_out_class'],
        'strategy': merged['setting'].map(strategy_map),
        'cv_mean_r': merged['cv_mean_r'],
        'test_pearson_r': merged['test_pearson_r'],
        'test_mae': merged['test_mae'],
    })
    print(f"Loaded {len(df)} rows from {CORRECTED_CV_CSV} + {SUMMARY_CSV}")
    return df


def load_signed_errors(results_root: Path) -> pd.DataFrame:
    """Per-sample signed errors (pred - true) of the retrained final models.

    Panel d aggregates these to fold means (bar heights) and overlays the raw
    values (scatter), so the near-unilateral class-wide bias direction stays
    visible rather than collapsing into a single mean per fold.
    """
    strategy_map = {'strict': 'Strict LOGO', 'improved': 'Few-shot LOGO'}
    preds = _load_per_sample_predictions(results_root)
    preds['strategy'] = preds['setting'].map(strategy_map)
    for strategy in strategy_map.values():
        for fold in CLASS_ORDER:
            mean_err = preds.loc[(preds['strategy'] == strategy) &
                                 (preds['fold'] == fold), 'signed_err'].mean()
            print(f"  {strategy:13} {fold:>4} mean signed err = {mean_err:.3f}")
    return preds[['fold', 'strategy', 'signed_err']]


def _load_per_sample_predictions(results_root: Path) -> pd.DataFrame:
    """Read the per-sample external-test predictions for panel d.

    Returns ``setting`` / ``fold`` / ``signed_err`` (pred - true, kcal mol^-1)
    columns. Prefers the release CSV ``data/logo_external_predictions.csv``
    (columns setting, held_out_class, motif, y_true, y_pred, signed_error),
    which the release build derives from the retrained-model predictions.json
    files. When that CSV is absent (working repository), the predictions.json
    files are parsed directly through ``logo_utils.load_logo_test_predictions``
    (same data as ``analysis/logo_signed_error_analysis.py``); that module is
    imported only on the fallback path so a released checkout does not need it.
    """
    if EXTERNAL_PRED_CSV.exists():
        preds = pd.read_csv(EXTERNAL_PRED_CSV)
        preds['signed_err'] = preds['y_pred'] - preds['y_true']
        return pd.DataFrame({
            'setting': preds['setting'],
            'fold': preds['held_out_class'],
            'signed_err': preds['signed_err'],
        })
    sys.path.insert(0, str(results_root))
    from logo_utils import load_logo_test_predictions  # working-repo fallback
    preds = load_logo_test_predictions(results_root)
    return pd.DataFrame({
        'setting': preds['setting'],
        'fold': preds['fold'],
        'signed_err': preds['signed_err'],
    })


def _style_panel_axes(ax, ylabel, ylim, show_xlabel):
    """Apply the shared five-class grouped-bar axis formatting."""
    x = np.arange(len(CLASS_ORDER))
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                       fontsize=PUBLICATION_FONTS['tick_label'])
    ax.set_xlim(-0.60, len(CLASS_ORDER) - 0.40)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=PUBLICATION_FONTS['axis_label'],
                  fontweight='bold')
    if show_xlabel:
        ax.set_xlabel('Target class', fontsize=PUBLICATION_FONTS['axis_label'],
                      fontweight='bold')
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    ax.grid(False)
    ax.set_axisbelow(True)


def _draw_paired_panel(ax, df, column, ylabel, ylim, show_xlabel=True):
    """Draw one grouped strict/few-shot bar panel (side-by-side bars)."""
    x = np.arange(len(CLASS_ORDER))
    offsets = {'Strict LOGO': -0.20, 'Few-shot LOGO': +0.20}
    # Bars use the soft fill (base hue blended toward white, matching the
    # Figure 4 violins); the mean guides keep the saturated base hues so the
    # dashed lines stay visible over the pale fills.
    fills = {'Strict LOGO': STRICT_FILL, 'Few-shot LOGO': FEWSHOT_FILL}
    means = {'Strict LOGO': STRICT_COLOR, 'Few-shot LOGO': FEWSHOT_COLOR}
    bar_width = 0.38

    for strategy, offset in offsets.items():
        vals = np.array([df.loc[(df['fold'] == c) & (df['strategy'] == strategy),
                                column].item() for c in CLASS_ORDER])
        ax.bar(x + offset, vals, width=bar_width, color=fills[strategy],
               edgecolor='black', linewidth=0.8, zorder=3)
        mean = vals.mean()
        ax.axhline(mean, color=means[strategy], linestyle='--',
                   linewidth=1.6, alpha=0.85, zorder=4)
        print(f"  {strategy} mean {column} = {mean:.3f}")

    _style_panel_axes(ax, ylabel, ylim, show_xlabel)


def _draw_signed_error_panel(ax, signed, show_xlabel=True):
    """Grouped strict/few-shot mean signed-error bars with per-sample scatter.

    Bar heights are the fold means (matching panels a-c in style); the
    jittered points are the per-sample residuals, which show that the
    class-level offsets are near-unilateral (e.g. every strict A/N sample is
    over-predicted) rather than averages of offsetting errors. Scatter style
    follows the project convention (jitter within the bar, alpha=0.5, s=35,
    black edges); a fixed seed keeps the jitter reproducible.
    """
    x = np.arange(len(CLASS_ORDER))
    offsets = {'Strict LOGO': -0.20, 'Few-shot LOGO': +0.20}
    fills = {'Strict LOGO': STRICT_FILL, 'Few-shot LOGO': FEWSHOT_FILL}
    points = {'Strict LOGO': STRICT_COLOR, 'Few-shot LOGO': FEWSHOT_COLOR}
    bar_width = 0.38
    rng = np.random.default_rng(42)

    for strategy, offset in offsets.items():
        vals = np.array([
            signed.loc[(signed['fold'] == c) & (signed['strategy'] == strategy),
                       'signed_err'].mean() for c in CLASS_ORDER])
        ax.bar(x + offset, vals, width=bar_width, color=fills[strategy],
               edgecolor='black', linewidth=0.8, zorder=3)
        for center, fold in zip(x + offset, CLASS_ORDER):
            sample = signed.loc[(signed['fold'] == fold) &
                                (signed['strategy'] == strategy), 'signed_err']
            ax.scatter(rng.uniform(center - 0.10, center + 0.10, len(sample)),
                       sample, s=35, color=points[strategy], alpha=0.5,
                       edgecolors='black', linewidths=0.4, zorder=4)

    # Zero reference: solid black so it reads distinctly from the dashed
    # five-class-mean guides used in panels a-c.
    ax.axhline(0.0, color='black', linewidth=1.6, zorder=5)
    _style_panel_axes(ax, SIGNED_ERR_YLABEL, SIGNED_ERR_YLIM, show_xlabel)


def create_logo_performance_figure(df: pd.DataFrame, signed_err: pd.DataFrame):
    """Create the four-panel 2x2 strict vs few-shot LOGO comparison figure.

    Panels a-c reuse the summary-CSV metrics (r/MAE, bars start at zero);
    panel d shows the mean signed error with a zero line. Only the bottom-row
    panels carry an x-axis label so the top row keeps its vertical space.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))

    for ax, (column, ylabel, ylim) in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                                          PANEL_SPECS):
        _draw_paired_panel(ax, df, column, ylabel, ylim,
                           show_xlabel=(ax is axes[1, 0]))
    _draw_signed_error_panel(axes[1, 1], signed_err, show_xlabel=True)

    handles = [
        mpatches.Patch(facecolor=STRICT_FILL, edgecolor='black',
                       linewidth=0.8, label='Strict LOGO'),
        mpatches.Patch(facecolor=FEWSHOT_FILL, edgecolor='black',
                       linewidth=0.8, label='Few-shot LOGO'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.6,
               label='Five-class mean'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               prop={'family': 'Times New Roman', 'size': PUBLICATION_FONTS['legend']},
               bbox_to_anchor=(0.5, -0.03))

    # Panel letters follow the Scientific Reports convention (bold lowercase),
    # larger than the axis-label font, offset left of the ylabel. The offset
    # is generous because the long ylabels of panels c/d extend into this
    # corner at the top of the axes.
    for ax, label in zip(axes.flat, ['a', 'b', 'c', 'd']):
        ax.text(-0.30, 1.02, label, transform=ax.transAxes,
                fontsize=42, fontweight='bold',
                fontfamily='Times New Roman',
                verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0.38, hspace=0.30)
    return fig, axes


def main() -> None:
    """Load the summary CSV and per-sample predictions, draw Figure 5, save PNG."""
    df = load_logo_summary()
    print("Panel means (reported in the manuscript text):")
    signed_err = load_signed_errors(RESULTS_ROOT)
    fig, _ = create_logo_performance_figure(df, signed_err)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, str(OUTPUT_PATH))
    print(f"  saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
