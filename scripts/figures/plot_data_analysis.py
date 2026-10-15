"""Generate data analysis figures for OP_unit technical report.

Creates the following figures:
- Fig 1.7: ΔG distribution of 120 building blocks (violin plot)
- Fig 1.8: Structural parameter correlations (scatter matrix)
- Fig 1.9: Representative PMF curves
- Fig 1.14: Predicted vs actual scatter plot
- Fig 1.15: High-throughput prediction heatmap
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from plot_utils import (
    ANION_COLORS,
    ANION_FULL_NAMES,
    ANION_LABELS,
    PUBLICATION_FONTS,
    center_rotated_xticklabels,
    format_axis_labels,
    get_anion_colors,
    get_publication_style,
    save_figure,
    save_publication_figure,
    setup_plot_style,
)
from public_data import (
    load_logo_external_predictions,
    load_logo_screening,
    load_reference_table,
)

FOLD_DIRS = ['fold_A_N', 'fold_C', 'fold_H', 'fold_P', 'fold_S']

# Manuscript six terminal-subgroup styling shared by Figure 2a/2b, S4 and S2:
# A = OH-terminated (hA) and N = O⁻-terminated (hN) — the A_N LOGO class
# split for descriptive analyses. Per the manuscript notation decision
# (2026-09-02, rev.), ALL figures use the single-letter codes: descriptive
# figures show A, N, C, H, P, S; ML/LOGO figures merge A and N into "A/N".
# Captions decode the codes; no chemical-formula tick labels anywhere.
SUBGROUP_ORDER = ['A', 'N', 'C', 'H', 'P', 'S']
SUBGROUP_COLORS = {
    'A': ANION_COLORS['A_N'],   # A keeps the ANION blue
    'N': '#6baed6',             # N gets a lighter sibling blue
    'C': ANION_COLORS['C'],
    'H': ANION_COLORS['H'],
    'P': ANION_COLORS['P'],
    'S': ANION_COLORS['S'],
}
SUBGROUP_LABELS = {code: code for code in SUBGROUP_ORDER}

# Panel-id -> manuscript class code for PMF panel titles (Figure 3 / S6).
CLASS_CODE_TITLES = {'hA': 'A', 'hN': 'N', 'C': 'C', 'H': 'H', 'P': 'P', 'S': 'S'}

# Lighter ±SD band for the manuscript Fig. 3 / S5 PMF figures: the curve
# colours there are fully saturated (matching the Fig. 2b points), so the
# default 0.2 band opacity muddies overlapping motifs. The shared
# report/patent figure keeps the historical 0.2 default.
MANUSCRIPT_PMF_FILL_ALPHA = 0.12

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = BASE_DIR / 'scripts'
DATA_DIR = BASE_DIR / 'data'
PMF_DATA_DIR = DATA_DIR / 'pmf_profiles'
def replicate_series(base: str, count: int = 5) -> list[str]:
    """Generate standard replicate filenames (without extension)."""
    return [f"{base}-{idx}" for idx in range(1, count + 1)]

PMF_PANEL_CONFIG = [
    {
        'id': 'hA',
        # Panel title uses short functional label: OH
        'title': r'$\mathrm{OH}$',
        'systems': {
            'hA001': replicate_series('hA001', 5),
            'hA005': replicate_series('hA005', 5),
            'hA021': replicate_series('hA021', 5),
        }
    },
    {
        'id': 'hN',
        # Panel title uses short functional label: O⁻
        'title': r'$\mathrm{O^{-}}$',
        'systems': {
            'hN001': replicate_series('hN001', 5),
            'hN005': replicate_series('hN005', 5),
            'hN021': replicate_series('hN021', 5),
        }
    },
    {
        # Carboxylate panel: use hC* heads to stay consistent
        # with the 120-molecule public reference table naming scheme.
        'id': 'C',
        'title': ANION_LABELS.get('C', 'C'),
        'systems': {
            'hC001': replicate_series('hC001', 5),
            'hC017': replicate_series('hC017', 5),
            'hC021': replicate_series('hC021', 5),
        }
    },
    {
        # Sulfonate panel: use hH* heads
        'id': 'H',
        'title': ANION_LABELS.get('H', 'H'),
        'systems': {
            'hH001': replicate_series('hH001', 5),
            'hH017': replicate_series('hH017', 6),
            'hH085': replicate_series('hH085', 5),
        }
    },
    {
        # Phosphate ester panel: updated with hP001 and hP021
        'id': 'P',
        'title': ANION_LABELS.get('P', 'P'),
        'systems': {
            'hP001': replicate_series('hP001', 5),
            'hP005': replicate_series('hP005', 5),
            'hP021': replicate_series('hP021', 5),
        }
    },
    {
        # Sulfate ester panel: updated with hS005 and hS021
        'id': 'S',
        'title': ANION_LABELS.get('S', 'S'),
        'systems': {
            'hS001': replicate_series('hS001', 5),
            'hS005': replicate_series('hS005', 5),
            'hS021': replicate_series('hS021', 5),
        }
    },
]

# Manuscript-only output directory (new Figure 3 / SI figures must not
# overwrite the shared common_figures used by the report and patent).
MANUSCRIPT_FIGURE_DIR = BASE_DIR / 'manuscript' / 'Figures'

# Fixed structural selection for Figure 3: the same three R-groups in every
# terminal-chemistry class (001 = methyl, 005 = ethyl, 021 = six-membered
# ring), so the six panels are directly comparable. R-groups not shown here
# (013 = iso-propyl, 017 = five-membered ring, 029 = oxygen-containing
# six-membered ring) are covered in Supplementary Fig. S2 instead.
# Exception: the H (sulfonate) panel uses hH069 (an ethyl with the longest
# sulfonate linker) in place of hH005, whose PMF replicates are anomalously
# noisy (SD ~1.9 kcal/mol around z = 2.3 nm).
FIXED_RGROUP_SYSTEMS = {
    'hA': ['hA001', 'hA005', 'hA021'],
    'hN': ['hN001', 'hN005', 'hN021'],
    'hC': ['hC001', 'hC005', 'hC021'],
    'hH': ['hH001', 'hH069', 'hH021'],
    'hP': ['hP001', 'hP005', 'hP021'],
    'hS': ['hS001', 'hS005', 'hS021'],
}

# Supplementary Fig. S2 selection: three further motifs per class covering the
# remaining R-groups, complementing the fixed selection of Figure 3.
ADDITIONAL_RGROUP_SYSTEMS = {
    'hA': ['hA013', 'hA017', 'hA029'],
    'hN': ['hN013', 'hN017', 'hN029'],
    'hC': ['hC013', 'hC017', 'hC029'],
    'hH': ['hH013', 'hH017', 'hH029'],
    'hP': ['hP013', 'hP017', 'hP029'],
    'hS': ['hS013', 'hS017', 'hS029'],
}


def _build_panel_config(systems_by_prefix: dict) -> list:
    """Build a PMF panel config with the given motifs per h-prefix class.

    Preserves the panel order and titles of ``PMF_PANEL_CONFIG`` while
    replacing the motif selection. Each motif keeps five replicate files.
    """
    panels = []
    for prefix, template in zip(PMF_PREFIX_ORDER, PMF_PANEL_CONFIG):
        panel = dict(template)
        panel['systems'] = {
            name: replicate_series(name, 5)
            for name in systems_by_prefix[prefix]
        }
        panels.append(panel)
    return panels

R_GROUP_ORDER = ['Methyl', 'Ethyl', 'Propyl', 'Isopropyl',
                 '5-Member Ring', '6-Member Ring', '7-Member Ring', 'Oxy-6-Member']
PMF_PREFIX_ORDER = ['hA', 'hN', 'hC', 'hH', 'hP', 'hS']
R_GROUP_MAP = {
    1: 'Methyl',
    5: 'Ethyl',
    9: 'Propyl',
    13: 'Isopropyl',
    17: '5-Member Ring',
    21: '6-Member Ring',
    25: '7-Member Ring',
    29: 'Oxy-6-Member'
}

# Manuscript eight-colour R-group palette (2026-09-03, shared by Figure 2c
# and the Figure 3 / Supplementary Fig. S2 PMF curve legends). Linear chains
# use a warm sand→orange→vermillion→brown progression and cyclic rings a
# cool bluish-green→sky-blue→blue→reddish-purple progression, so the two
# R-group families read as separate colour blocks while the individual hues
# stay in the same muted register as the tab10-based anion palette. Keys are
# the exact strings produced by ``classify_r_group`` / ``R_GROUP_ORDER``.
R_GROUP_COLORS = {
    'Methyl': '#BFA94E',          # deeper sand / ochre (linear)
    'Ethyl': '#E69F00',           # orange (linear)
    'Propyl': '#D55E00',          # vermillion (linear)
    'Isopropyl': '#8C510A',       # brown (linear)
    '5-Member Ring': '#009E73',   # bluish green (cyclic)
    '6-Member Ring': '#56B4E9',   # sky blue (cyclic)
    '7-Member Ring': '#0072B2',   # blue (cyclic)
    'Oxy-6-Member': '#CC79A7',    # reddish purple (cyclic)
}

# Short tick / legend labels for the eight R groups (Figure 2c x-axis and
# the PMF curve legends of Figure 3 / Supplementary Fig. S2).
R_GROUP_SHORT_LABELS = {
    'Methyl': 'Me',
    'Ethyl': 'Et',
    'Propyl': 'n-Pr',
    'Isopropyl': 'i-Pr',
    '5-Member Ring': '5-Ring',
    '6-Member Ring': '6-Ring',
    '7-Member Ring': '7-Ring',
    'Oxy-6-Member': 'Oxy-6',
}


def pmf_legend_label(system_name: str) -> str:
    """Return legend label for a PMF system.

    保留h-前缀表示head分子类型。
    去掉重复编号后缀（如-3, -1等）。

    示例:
    - "hC001-3" → "hC001"
    - "hH005-1" → "hH005"
    - "C055-3" → "C055" (非h-prefixed的tail分子保持原样)
    """
    base = system_name.split('-', 1)[0]
    return base  # 直接返回，保留h前缀


def classify_r_group(name):
    if pd.isna(name):
        return 'Other'
    try:
        idx = int(str(name)[2:])
    except ValueError:
        return 'Other'
    base = ((idx - 1) % 32) + 1
    return R_GROUP_MAP.get(base, 'Other')


def load_motif_smiles() -> dict[str, str]:
    """Return a Name -> SMILES mapping for the public reference table."""
    df = load_reference_table()
    return dict(zip(df['Name'], df['SMILES']))


def _render_motif_structure(smiles: str, width_px: int = 600, height_px: int = 460):
    """Render one motif's 2D structure as an RGB array, Fig. 6 style.

    Mirrors ``render_smiles`` in ``manuscript/plot_fig6_prospective.py`` so
    the two structure figure sets look identical: uniform bond lengths (the
    2D conformer is rescaled to a fixed coordinate span), fixed-pixel atom
    labels, and RDKit's standard element colouring. Structures are drawn
    from the SMILES in the public reference table, exactly the molecules whose curves
    appear in the panels.

    Args:
        smiles: SMILES string of the motif.
        width_px: Pixel width of the rendered image.
        height_px: Pixel height of the rendered image.

    Returns:
        (height, width, 3) float array in [0, 1], or None if SMILES parsing fails.
    """
    from io import BytesIO

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)

    # Uniform scale: normalise the conformer to a fixed coordinate span so
    # bond lengths match across molecules of different sizes (same approach
    # and target span as the Fig. 6 structure grid).
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span > 0.0:
        scale = 24.0 / span
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D((p.x - cx) * scale, (p.y - cy) * scale, 0.0))

    drawer = rdMolDraw2D.MolDraw2DCairo(width_px, height_px)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 5
    opts.padding = 0.12
    # fixedFontSize (exact pixels) because baseFontSize is non-monotonic in
    # this rdkit build; see plot_fig6_prospective.render_smiles.
    # Larger atom labels than the Fig. 6 default (62 vs 50 px): the Fig. 3
    # strips are viewed smaller, so N/O/P/S symbols need to stay legible;
    # bond width bumped 4 -> 5 to keep the line/label proportion.
    opts.fixedFontSize = 62
    opts.additionalAtomLabelPadding = 0.2
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return plt.imread(BytesIO(drawer.GetDrawingText()))


def _draw_structure_strip(ax, motif_names, smiles_map):
    """Draw a row of 2D structures with motif-ID labels inside ``ax``.

    The strip sits in its own frameless axes above a PMF panel; structures
    are centred in equal slots, ordered exactly like the panel's legend
    entries (the legend's Me/Et/6-Ring labels identify them, so no
    per-structure text is drawn), following the Fig. 6a style.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    if not motif_names:
        return
    for k, name in enumerate(motif_names):
        smiles = smiles_map.get(name)
        if not smiles:
            continue
        img = _render_motif_structure(smiles)
        if img is None:
            continue
        slot_centre = (2 * k + 1) / (2 * len(motif_names))
        image = OffsetImage(img)
        image.set_zoom(0.185)
        # Anchor the structure low in the strip axes so the block sits
        # tight against the PMF panel below it (the strip belongs to that
        # panel, not to whatever is above).
        annotation = AnnotationBbox(image, (slot_centre, 0.25), frameon=False)
        ax.add_artist(annotation)


def load_retrained_predictions(logo_type: str = 'improved') -> pd.DataFrame:
    df = load_logo_external_predictions()
    return df[df['setting'] == logo_type][
        ['held_out_class', 'y_true', 'y_pred']
    ].rename(columns={'held_out_class': 'fold'})


def load_logo_scores(results_dir='strict'):
    """Load public LOGO screening scores for fold-level analysis."""
    df = load_logo_screening()
    if results_dir.endswith('improved'):
        df = df[df['setting'].eq('Few-shot LOGO')]
    else:
        df = df[df['setting'].eq('Strict LOGO')]
    return df.rename(columns={'internal_validation_pearson_r': 'score'})

def load_molecular_data():
    """Load molecular data from the public 120-motif table."""
    df = load_reference_table()
    print(f"Loaded {len(df)} molecules from the public reference table")

    # Extract anion group from code prefix (e.g., hA, hN, hC...)
    def extract_anion_group(name):
        if pd.isna(name):
            return 'Unknown'
        name = str(name).upper()
        if len(name) < 2:
            return 'Unknown'
        prefix = name[1]
        if prefix in ('A', 'N'):
            return 'A_N'
        elif prefix == 'C':
            return 'C'
        elif prefix == 'H':
            return 'H'
        elif prefix == 'P':
            return 'P'
        elif prefix == 'S':
            return 'S'
        else:
            return 'Unknown'

    df['anion_group'] = df['Name'].apply(extract_anion_group)

    return df


def compute_structure_descriptors(smiles):
    """Compute the shared structure descriptors for one molecule.

    This is the single descriptor code path for every figure: the combined
    data analysis figure (Fig 1.3) and the manuscript Supplementary Fig. S4
    both call this function, so their per-molecule values are identical by
    construction.

    Returns:
        tuple: (mol_wt, logP, Labute ASA, max |Gasteiger charge|, n rotatable bonds)
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    if pd.isna(smiles):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Compute 2D descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    asa = rdMolDescriptors.CalcLabuteASA(mol)

    # Compute 3D-dependent descriptors (use 2D approximations)
    # MaxPartialCharge: max |Gasteiger partial charge| over atoms,
    # a charge-localization descriptor (NOT a dipole moment / dipole proxy)
    try:
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        max_charge = max(abs(c) for c in charges if not np.isnan(c))
    except:
        max_charge = np.nan

    # Number of rotatable bonds (flexibility proxy)
    n_rotatable = Descriptors.NumRotatableBonds(mol)

    return mw, logp, asa, max_charge, n_rotatable


def create_combined_data_figure(df):
    """Create combined 3×2 data analysis figure (Fig 1.3).

    Layout: 3 rows × 2 columns (vertical orientation), legend on right

    Panels:
        A (row 1, left): Anion type vs ΔG distribution (violin plot)
        B (row 1, right): Molecular weight vs ΔG (scatter)
        C (row 2, left): R-group vs mean ΔG (bar chart)
        D (row 2, right): logP vs ΔG (scatter)
        E (row 3, left): MaxPartialCharge vs ΔG (scatter) [charge-localization descriptor]
        F (row 3, right): Labute ASA vs ΔG (scatter)

    Returns:
        tuple: (fig, axes) where axes is a 3x2 array of Axes objects

    """
    from rdkit import Chem

    df = df.copy()
    df['r_group'] = df['Name'].apply(classify_r_group)

    desc = df['SMILES'].apply(compute_structure_descriptors)
    df[['mol_wt', 'logP', 'ASA', 'max_charge', 'n_rotatable']] = pd.DataFrame(desc.tolist(), index=df.index)

    # Create 3×2 subplot layout (3 rows, 2 columns) - vertical orientation
    style = get_publication_style()
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        ax_violin, ax_mw, ax_bar, ax_logp, ax_charge, ax_asa = axes.flatten()

        # --- Panel A: Anion type violin plot ---
        order = ['A_N', 'C', 'H', 'P', 'S']
        sns.violinplot(data=df, x='anion_group', y='dG_rel_kcal_per_mol', hue='anion_group',
                       order=order, palette=ANION_COLORS, ax=ax_violin,
                       dodge=False, legend=False, cut=0)
        sns.stripplot(data=df, x='anion_group', y='dG_rel_kcal_per_mol',
                      order=order, color='black', alpha=0.3,
                      size=3, jitter=True, ax=ax_violin)
        ax_violin.set_xlabel('Anion Type', fontfamily='Times New Roman',
                             fontsize=PUBLICATION_FONTS['axis_label'])
        ax_violin.set_ylabel('ΔG (kcal/mol)', fontfamily='Times New Roman',
                             fontsize=PUBLICATION_FONTS['axis_label'])
        ax_violin.set_xticks(range(len(order)))
        # Use short labels (A_N, C, H, P, S) instead of chemical formulas
        ax_violin.set_xticklabels(order, rotation=0)
        ax_violin.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])

        # --- Panel B: R-group bar chart ---
        r_stats = df.groupby('r_group')['dG_rel_kcal_per_mol'].agg(['mean', 'std']).reindex(R_GROUP_ORDER)
        ax_bar.barh(r_stats.index, r_stats['mean'],
                    xerr=r_stats['std'], color='#2ca02c', alpha=0.85,
                    ecolor='black', capsize=5)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Mean ΔG (kcal/mol)', fontfamily='Times New Roman',
                          fontsize=PUBLICATION_FONTS['axis_label'])
        # No ylabel per user request

        # Simplify R-group labels
        r_group_short = {
            'Methyl': 'Me',
            'Ethyl': 'Et',
            'Propyl': 'n-Pr',
            'Isopropyl': 'i-Pr',
            '5-Member Ring': '5-Ring',
            '6-Member Ring': '6-Ring',
            '7-Member Ring': '7-Ring',
            'Oxy-6-Member': 'Oxy-6'
        }
        ax_bar.set_yticklabels([r_group_short.get(label, label) for label in r_stats.index],
                               fontfamily='Times New Roman')
        ax_bar.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])

        # --- Helper function for scatter panels ---
        def _scatter_panel(ax, x, y, xlabel, ylabel='ΔG (kcal/mol)'):
            colors = df['anion_group'].map(ANION_COLORS)
            ax.scatter(x, y, c=colors, alpha=0.75, s=40, edgecolor='none')
            ax.set_xlabel(xlabel, fontfamily='Times New Roman',
                          fontsize=PUBLICATION_FONTS['axis_label'])
            ax.set_ylabel(ylabel, fontfamily='Times New Roman',
                          fontsize=PUBLICATION_FONTS['axis_label'])
            ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])

        # --- Panel C: Molecular weight scatter ---
        _scatter_panel(ax_mw, df['mol_wt'], df['dG_rel_kcal_per_mol'],
                       xlabel='Molecular Weight (Da)')

        # --- Panel D: logP scatter ---
        _scatter_panel(ax_logp, df['logP'], df['dG_rel_kcal_per_mol'],
                       xlabel='logP')

        # --- Panel E: MaxPartialCharge scatter [charge-localization descriptor] ---
        _scatter_panel(ax_charge, df['max_charge'], df['dG_rel_kcal_per_mol'],
                       xlabel='Max Partial Charge (|e|)')

        # --- Panel F: Labute ASA scatter ---
        _scatter_panel(ax_asa, df['ASA'], df['dG_rel_kcal_per_mol'],
                       xlabel='Labute ASA (Å²)')

        # --- Add panel labels A-F (top-left corner) ---
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for ax, label in zip(axes.flatten(), panel_labels):
            ax.text(-0.08, 1.02, label,
                    transform=ax.transAxes,
                    fontsize=PUBLICATION_FONTS['title'],
                    fontweight='bold',
                    fontfamily='Times New Roman',
                    verticalalignment='bottom',
                    horizontalalignment='right')

        # --- Add unified legend (bottom center, below panels) ---
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   label=ANION_LABELS.get(group, group),
                   markerfacecolor=ANION_COLORS.get(group, '#808080'),
                   markersize=10)
            for group in ['A_N', 'C', 'H', 'P', 'S']
        ]
        fig.legend(handles=legend_handles,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.05),
                   ncol=5,  # Display horizontally (5 anion types)
                   frameon=True,
                   fontsize=PUBLICATION_FONTS['legend'],
                   edgecolor='black',
                   fancybox=True,
                   title='Anion Type',
                   title_fontsize=PUBLICATION_FONTS['legend'] + 2)

        # Apply axis styling
        for ax in axes.flatten():
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            for spine in ax.spines.values():
                spine.set_linewidth(style.get('axes.linewidth', 2.0))

        fig.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for legend at bottom

    return fig, axes

def create_deltaG_distribution_figure(df):
    """Create ΔG distribution violin plot (Fig 1.7)."""
    fig, ax = setup_plot_style(figsize=(12, 8), panel_type='distribution')

    order = ['A_N', 'C', 'H', 'P', 'S']

    # Create violin plot by anion group with explicit hue to avoid palette warnings
    sns.violinplot(data=df, x='anion_group', y='dG_rel_kcal_per_mol', hue='anion_group',
                   order=order, palette=ANION_COLORS, ax=ax,
                   dodge=False, legend=False, cut=0)

    # Add individual data points (simplified for publication)
    sns.stripplot(data=df, x='anion_group', y='dG_rel_kcal_per_mol',
                  order=order, color='black', alpha=0.3,
                  size=3, jitter=True, ax=ax)

    # Publication styling - English labels only, no title
    ax.set_xlabel('Anion Type', fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('ΔG (kcal/mol)', fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('')  # No title for publication

    # Use chemical formulas with proper Unicode subscripts/superscripts
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ANION_LABELS.get(g, g) for g in order], rotation=0)

    plt.tight_layout()
    return fig, ax

def create_structural_correlation_matrix(df):
    """Create 2x2 structural analysis figure (Fig 1.4).

    Panels (no titles inside; meanings described in caption):
        (0, 0): R-group vs mean ΔG (bar, 带误差线)
        (0, 1): Molecular weight vs ΔG (scatter)
        (1, 0): logP vs ΔG (scatter)
        (1, 1): Labute ASA vs ΔG (scatter, 连续表面积类描述符)
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    df = df.copy()
    df['r_group'] = df['Name'].apply(classify_r_group)

    # 计算分子量、logP、TPSA 和 Labute ASA
    def _compute_desc(smiles):
        if pd.isna(smiles):
            return np.nan, np.nan, np.nan, np.nan
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.nan, np.nan, np.nan, np.nan
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        asa = rdMolDescriptors.CalcLabuteASA(mol)
        return mw, logp, tpsa, asa

    desc = df['SMILES'].apply(_compute_desc)
    df[['mol_wt', 'logP', 'TPSA', 'ASA']] = pd.DataFrame(desc.tolist(), index=df.index)

    style = get_publication_style()
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_bar, ax_mw, ax_logp, ax_asa = axes.flatten()

        # (0,0) R-group 平均 ΔG 条形图
        r_stats = df.groupby('r_group')['dG_rel_kcal_per_mol'].agg(['mean', 'std']).reindex(R_GROUP_ORDER)
        ax_bar.barh(
            r_stats.index,
            r_stats['mean'],
            xerr=r_stats['std'],
            color='#2ca02c',
            alpha=0.85,
            ecolor='black',
            capsize=5,
        )
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel(
            'Mean ΔG (kcal/mol)',
            fontfamily='Times New Roman',
            fontsize=PUBLICATION_FONTS['axis_label'],
        )

        # 通用散点绘制助手：只用坐标轴说明，不在子图内部写文字
        def _scatter_panel(ax, x, y, xlabel, ylabel='ΔG (kcal/mol)'):
            colors = df['anion_group'].map(ANION_COLORS)
            ax.scatter(
                x,
                y,
                c=colors,
                alpha=0.75,
                s=40,
                edgecolor='none',
            )
            ax.set_xlabel(
                xlabel,
                fontfamily='Times New Roman',
                fontsize=PUBLICATION_FONTS['axis_label'],
            )
            ax.set_ylabel(
                ylabel,
                fontfamily='Times New Roman',
                fontsize=PUBLICATION_FONTS['axis_label'],
            )

        # (0,1) 分子量 vs ΔG
        _scatter_panel(
            ax_mw,
            df['mol_wt'],
            df['dG_rel_kcal_per_mol'],
            xlabel='Molecular Weight (Da)',
        )

        # (1,0) logP vs ΔG
        _scatter_panel(
            ax_logp,
            df['logP'],
            df['dG_rel_kcal_per_mol'],
            xlabel='logP',
        )

        # (1,1) ASA vs ΔG
        _scatter_panel(
            ax_asa,
            df['ASA'],
            df['dG_rel_kcal_per_mol'],
            xlabel='Labute ASA (Å²)',
        )

        # 统一图例：放在右下角 ASA 面板中
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                label=ANION_LABELS.get(group, group),
                markerfacecolor=ANION_COLORS.get(group, '#808080'),
                markersize=8,
            )
            for group in ['A_N', 'C', 'H', 'P', 'S']
        ]
        ax_asa.legend(
            handles=legend_handles,
            loc='lower right',
            frameon=True,
            fontsize=PUBLICATION_FONTS['legend'] - 4,
            edgecolor='black',
            framealpha=0.9,
            fancybox=True,
        )

        # Tick 字体和边框线一并统一，字号稍微小一点，避免遮挡散点
        for axis in axes.flatten():
            axis.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'] - 4)
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            for spine in axis.spines.values():
                spine.set_linewidth(style.get('axes.linewidth', 2.0))

        fig.tight_layout()

    return fig, ax_bar

def _draw_pmf_panels(config, panel_titles=None, rgroup_colors: bool = False,
                     fill_alpha: float = 0.2, align_replicates: bool = False,
                     pmf_ylabel: bool = False,
                     structure_motifs: Optional[dict[str, list[str]]] = None):
    """Shared six-panel PMF plotting routine (mean curve plus ±SD band).

    Used by Figure 3 and Supplementary Fig. S2 with different motif
    selections. Each curve is the mean over the replicate PMFs and the
    shaded band is the standard deviation across them.

    Args:
        config: panel selection, as built by ``_build_panel_config``.
        structure_motifs: optional mapping of h-prefix class to the motif
            names drawn in that panel. When given (manuscript Fig. 3 mode),
            a frameless strip showing each motif's 2D structure (rendered
            in the same style as the Fig. 6 structure grid) is added above
            the corresponding PMF panel, so readers can see the exact
            molecules behind the curves. The strip order matches the panel
            legend order.
        panel_titles: optional mapping of panel id to a replacement title.
            The manuscript figures pass the single-letter class codes
            (A/N/C/H/P/S) here; when None the config's own titles
            (chemical formulas, used by the shared report/patent figure)
            are kept unchanged.
        rgroup_colors: when True, colour each curve with the manuscript
            eight-colour R-group palette (``R_GROUP_COLORS``) keyed by the
            system name and label the legend entry with the short R-group
            name (Me, Et, 6-Ring, ...). The default False keeps the
            historical Dark2 colours and h-prefixed motif identifiers so
            the shared report/patent figure reproduces unchanged.
        fill_alpha: opacity of the ±SD band. The default 0.2 reproduces
            the shared report/patent figure; the manuscript Fig. 3 / S5
            modes pass a lighter value so overlapping bands stay readable.
        align_replicates: when True, reference every replicate to zero in
            the far-field plateau (mean over the last 0.5 nm) before
            averaging, so the ±SD band reflects sampling variability only.
            WHAM PMFs are defined up to an arbitrary constant and the
            charged oxide-terminated (hN) runs do not share a common
            reference, so without this step their arbitrary offsets enter
            the band as spurious spread. The default False keeps the
            historical averaging for the shared report/patent figure.
        pmf_ylabel: when True, label the y-axis ``PMF`` instead of the
            historical ``ΔG``. The curve is the potential of mean force
            G(z) itself, while ΔG in this manuscript is reserved for
            state differences (ΔG_rel), so the manuscript Fig. 3 / S5
            modes pass True (2026-09-12 decision). The default False
            keeps the historical label for the shared report/patent
            figure.
    """
    if panel_titles is None:
        panel_titles = {}

    structure_smiles = load_motif_smiles() if structure_motifs else {}

    def load_xvg(file_path: Path):
        xs, ys = [], []
        if not file_path.exists():
            return None
        with open(file_path) as f:
            for line in f:
                if line.startswith(('#', '@')):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
                    except ValueError:
                        continue
        if not xs:
            return None
        x = np.array(xs)
        y = np.array(ys)
        # Ensure values are sorted for interpolation safety
        sort_idx = np.argsort(x)
        return x[sort_idx], y[sort_idx]

    def average_replicates(replicate_files):
        """Average multiple PMF replicates for a single system."""
        loaded = []
        for fname in replicate_files:
            data = load_xvg(PMF_DATA_DIR / f"{fname}.xvg")
            if data is not None:
                loaded.append(data)

        if not loaded:
            return None

        # Determine a common grid across all replicates (overlapping domain only)
        min_start = max(curve[0][0] for curve in loaded)
        max_end = min(curve[0][-1] for curve in loaded)
        if max_end <= min_start:
            reference_x = loaded[0][0]
        else:
            num_points = min(len(curve[0]) for curve in loaded)
            reference_x = np.linspace(min_start, max_end, num=num_points)

        aligned_curves = []
        for x_vals, y_vals in loaded:
            aligned = np.interp(reference_x, x_vals, y_vals)
            aligned_curves.append(aligned)

        stacked = np.vstack(aligned_curves)
        if align_replicates:
            # Reference each replicate to zero over the far-field plateau
            # (last 0.5 nm of the common grid) before aggregating, per the
            # G(4.0 nm) = 0 convention stated in the manuscript Methods.
            plateau = reference_x >= reference_x[-1] - 0.5
            stacked = stacked - stacked[:, plateau].mean(axis=1, keepdims=True)
        mean_curve = stacked.mean(axis=0)
        # Sample SD (ddof=1) for the manuscript figures: the five replicates
        # are a sample of the repeat distribution, and the label-uncertainty
        # statistics quoted in the main text / Table S3 (median 0.59) use the
        # sample SD. The shared report/patent PMF figure stays on population SD
        # (ddof=0) to keep its band width unchanged, matching the figures.md
        # isolation rule for report/patent shared figures.
        std_curve = stacked.std(axis=0, ddof=1 if pmf_ylabel else 0)

        # Shift PMF so that the potential at the farthest sampled
        # distance (proxy for “infinite” separation) is zero.
        # This is purely a plotting convention and does not alter
        # the relative shape or barriers of the PMF.
        if mean_curve.size > 0:
            mean_curve = mean_curve - mean_curve[-1]

        return reference_x, mean_curve, std_curve

    style = get_publication_style()
    with plt.rc_context(style):
        if structure_motifs is None:
            fig, axes_grid = plt.subplots(2, 3, figsize=(18, 10),
                                          sharex=False, sharey=False)
            axes = list(axes_grid.flatten())
            struct_axes = {}
        else:
            # Manuscript Fig. 3 mode: interleave a structure strip row above
            # each PMF row so every panel carries its own molecules. The
            # gridspec geometry is explicit (no tight_layout re-flow) so each
            # PMF axes keeps the same box height as the original 2x3 figure;
            # the strip rows add ~1.5 in of structure + label height only.
            fig = plt.figure(figsize=(18, 15.9))
            spec = fig.add_gridspec(4, 3,
                                    height_ratios=[0.34, 1.0, 0.34, 1.0],
                                    left=0.082, right=0.985, top=0.955, bottom=0.06,
                                    hspace=0.30, wspace=0.24)
            axes, struct_axes = [], {}
            for idx in range(len(config)):
                row, col = (0, idx) if idx < 3 else (2, idx - 3)
                struct_axes[idx] = fig.add_subplot(spec[row, col])
                axes.append(fig.add_subplot(spec[row + 1, col]))

        for idx, panel in enumerate(config):
            ax = axes[idx]
            systems = panel.get('systems', {})
            system_items = list(systems.items())
            colors = plt.cm.Dark2(np.linspace(0, 1, max(1, len(system_items))))

            has_data = False
            for (system_name, replicate_files), default_color in zip(system_items, colors):
                averaged = average_replicates(replicate_files)
                if averaged is None:
                    continue

                if rgroup_colors:
                    r_group = classify_r_group(system_name)
                    color = R_GROUP_COLORS.get(r_group, default_color)
                    legend_label = R_GROUP_SHORT_LABELS.get(
                        r_group, pmf_legend_label(system_name))
                else:
                    color = default_color
                    legend_label = pmf_legend_label(system_name)

                x_vals, mean_vals, std_vals = averaged
                ax.plot(
                    x_vals,
                    mean_vals,
                    label=legend_label,
                    linewidth=3.0,
                    color=color,
                )
                if len(replicate_files) > 1 and np.any(std_vals > 0):
                    ax.fill_between(
                        x_vals,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=color,
                        alpha=fill_alpha,
                        linewidth=0,
                    )
                has_data = True

            ax.set_title(
                panel_titles.get(panel.get('id', ''),
                                 panel.get('title', panel.get('id', ''))),
                fontsize=PUBLICATION_FONTS['subtitle'],
                fontfamily='Times New Roman',
                fontweight='bold',
            )
            ax.set_xlabel('z (nm)', fontfamily='Times New Roman')
            # Tighter label pad in the structure-strip mode: the strip layout
            # fixes the left margin explicitly (no tight_layout re-flow), and
            # the global axes.labelpad pushes the rotated ylabel past the
            # canvas edge there.
            ylabel_pad = 2 if structure_motifs is not None else None
            ylabel_text = (r'PMF ($\mathrm{kcal\cdot mol^{-1}}$)'
                           if pmf_ylabel else
                           r'$\Delta G$ ($\mathrm{kcal\cdot mol^{-1}}$)')
            if ylabel_pad is not None:
                ax.set_ylabel(ylabel_text,
                              fontfamily='Times New Roman', labelpad=ylabel_pad)
            else:
                ax.set_ylabel(ylabel_text,
                              fontfamily='Times New Roman')
            ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
            ax.grid(False)

            if has_data:
                # Place legend inside the panel so the legend frame
                # stays within the main plotting box, consistent with
                # earlier styling and other figures.
                ax.legend(
                    frameon=True,
                    fontsize=PUBLICATION_FONTS['legend'],
                    loc='upper right',
                    borderaxespad=0.3,
                    prop={'family': 'Times New Roman', 'size': PUBLICATION_FONTS['legend']},
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    'No Data',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=PUBLICATION_FONTS['annotation'],
                    fontfamily='Times New Roman',
                )

            if idx in struct_axes:
                # Strip order and tinting mirror the panel legend order.
                _draw_structure_strip(
                    struct_axes[idx], structure_motifs.get(PMF_PREFIX_ORDER[idx], []),
                    structure_smiles)

        total_axes = len(axes)
        for j in range(len(config), total_axes):
            axes[j].axis('off')

        # Legends are now inside each panel, so we can use a
        # standard tight_layout without reserving extra right margin.
        # The structure-strip mode sets its own explicit gridspec geometry
        # and must not be re-flowed (tight_layout would shrink the PMF axes).
        if structure_motifs is None:
            fig.tight_layout()

    return fig, axes[0]


def create_pmf_curves_figure(selection: str = 'original',
                             class_code_titles: bool = False,
                             rgroup_colors: bool = False,
                             fill_alpha: float = 0.2,
                             align_replicates: bool = False,
                             pmf_ylabel: bool = False,
                             structure_motifs: Optional[dict[str, list[str]]] = None):
    """Create the Figure 3 multi-panel PMF curves from real simulations.

    Args:
        selection: 'original' reproduces the historical PMF panel selection;
            'fixed-rgroup' uses the same three R-groups (methyl, ethyl,
            six-membered ring) in every terminal-chemistry class so the six
            panels are directly comparable. The original selection stays the
            default so the historical figure can always be reproduced.
        class_code_titles: when True, panel titles use the manuscript
            single-letter class codes (A, N, C, H, P, S) instead of the
            config's chemical-formula titles. Only the manuscript figure
            mode passes True; the shared report/patent figure keeps the
            original titles.
        rgroup_colors: when True, colour the curves with the manuscript
            eight-colour R-group palette and label the legends with the
            short R-group names (see ``_draw_pmf_panels``). Only the
            manuscript figure mode passes True.
        fill_alpha: opacity of the ±SD band (see ``_draw_pmf_panels``).
            Only the manuscript figure mode passes a non-default value.
        align_replicates: far-field referencing of each replicate before
            averaging (see ``_draw_pmf_panels``). Only the manuscript
            figure mode passes True.
        pmf_ylabel: label the y-axis ``PMF`` instead of ``ΔG``
            (see ``_draw_pmf_panels``). Only the manuscript figure mode
            passes True.
        structure_motifs: motif names per h-prefix class for the structure
            strips above each panel (see ``_draw_pmf_panels``). Only the
            manuscript fixed-rgroup mode passes FIXED_RGROUP_SYSTEMS.
    """
    config = (
        _build_panel_config(FIXED_RGROUP_SYSTEMS)
        if selection == 'fixed-rgroup'
        else PMF_PANEL_CONFIG
    )
    if structure_motifs is None and selection == 'fixed-rgroup':
        structure_motifs = FIXED_RGROUP_SYSTEMS
    return _draw_pmf_panels(
        config,
        panel_titles=CLASS_CODE_TITLES if class_code_titles else None,
        rgroup_colors=rgroup_colors,
        fill_alpha=fill_alpha,
        align_replicates=align_replicates,
        pmf_ylabel=pmf_ylabel,
        structure_motifs=structure_motifs,
    )


def create_supplementary_pmf_figure(class_code_titles: bool = False,
                                    rgroup_colors: bool = False,
                                    fill_alpha: float = 0.2,
                                    align_replicates: bool = False,
                                    pmf_ylabel: bool = False):
    """Create Supplementary Fig. S2: additional representative PMF profiles.

    Shows three further motifs per terminal-chemistry class (iso-propyl,
    five-membered-ring and oxygen-containing six-membered-ring analogues),
    complementing the fixed R-group selection of Figure 3 and demonstrating
    that the class-dependent PMF shapes are robust across R-groups.

    Args:
        class_code_titles: same meaning as in ``create_pmf_curves_figure``.
        rgroup_colors: same meaning as in ``create_pmf_curves_figure``.
        fill_alpha: same meaning as in ``create_pmf_curves_figure``.
        align_replicates: same meaning as in ``create_pmf_curves_figure``.
        pmf_ylabel: same meaning as in ``create_pmf_curves_figure``.
    """
    return _draw_pmf_panels(
        _build_panel_config(ADDITIONAL_RGROUP_SYSTEMS),
        panel_titles=CLASS_CODE_TITLES if class_code_titles else None,
        rgroup_colors=rgroup_colors,
        fill_alpha=fill_alpha,
        align_replicates=align_replicates,
        pmf_ylabel=pmf_ylabel,
    )


def create_prediction_scatter_figure(pred_df):
    """Create predicted vs actual scatter plot (Fig 1.14) using retrained models."""
    if pred_df.empty:
        raise ValueError("No retrained prediction data available")

    fig, ax = setup_plot_style(figsize=(10, 10))
    actual = pred_df['y_true'].values
    predicted = pred_df['y_pred'].values
    folds = pred_df['fold'].values

    colors = [ANION_COLORS.get(fold, '#808080') for fold in folds]
    ax.scatter(actual, predicted, c=colors, alpha=0.75, s=60, edgecolor='none')

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

    from sklearn.metrics import mean_absolute_error, r2_score
    pearson = np.corrcoef(actual, predicted)[0, 1]
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    format_axis_labels(ax,
                     xlabel='Actual ΔG (kcal/mol)',
                     ylabel='Predicted ΔG (kcal/mol)',
                     title=None)

    ax.set_aspect('equal')
    ax.grid(False)

    stats_text = f"Pearson $r$ = {pearson:.3f}\n$R^2$ = {r2:.3f}\nMAE = {mae:.2f}"
    ax.text(0.02, 0.05, stats_text, transform=ax.transAxes,
            fontsize=PUBLICATION_FONTS['annotation'], fontfamily='Times New Roman',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=ANION_LABELS.get(group, group),
               markerfacecolor=ANION_COLORS.get(group, '#808080'), markersize=10)
        for group in ['A_N', 'C', 'H', 'P', 'S']
    ]
    ax.legend(handles=legend_handles, loc='lower right', frameon=True,
              fontsize=PUBLICATION_FONTS['legend'])

    plt.tight_layout()
    return fig, ax

def create_highthroughput_heatmap(df):
    """Create heatmap of observed ΔG grouped by R group and anion."""
    fig, ax = setup_plot_style(figsize=(14, 10))

    df = df.copy()
    df['r_group'] = df['Name'].apply(classify_r_group)

    # Separate A and N groups for better visualization
    def extract_anion_subgroup(name):
        if pd.isna(name):
            return 'Unknown'
        name = str(name).upper()
        if len(name) < 2:
            return 'Unknown'
        prefix = name[1]
        # Separate hA (-OH) and hN (O⁻)
        if prefix == 'A':
            return 'A'
        elif prefix == 'N':
            return 'N'
        elif prefix == 'C':
            return 'C'
        elif prefix == 'H':
            return 'H'
        elif prefix == 'P':
            return 'P'
        elif prefix == 'S':
            return 'S'
        else:
            return 'Unknown'

    df['anion_subgroup'] = df['Name'].apply(extract_anion_subgroup)
    pivot = df.pivot_table(values='dG_rel_kcal_per_mol', index='r_group', columns='anion_subgroup', aggfunc='mean')
    pivot = pivot.reindex(R_GROUP_ORDER)

    # Separate -OH and O⁻ in the column order
    anion_order = ['A', 'N', 'C', 'H', 'P', 'S']
    pivot = pivot[[col for col in anion_order if col in pivot.columns]]

    heat = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=df['dG_rel_kcal_per_mol'].mean(),
                       annot_kws={'fontsize': PUBLICATION_FONTS['value_text'], 'fontfamily': 'Times New Roman'},
                       square=True, cbar_kws={'label': 'Mean ΔG (kcal/mol)'}, ax=ax)

    if heat.collections:
        cbar = heat.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=PUBLICATION_FONTS['colorbar'])
            for label in cbar.ax.get_yticklabels():
                label.set_fontfamily('Times New Roman')
            cbar.ax.set_ylabel('Mean ΔG (kcal/mol)',
                               fontsize=PUBLICATION_FONTS['colorbar'], fontweight='bold',
                               family='Times New Roman')

    # Use simple code labels for x-axis (matching Table 1.2 fold labels)
    anion_label_map = {
        'A': 'A',
        'N': 'N',
        'C': 'C',
        'H': 'H',
        'P': 'P',
        'S': 'S'
    }
    ax.set_xticklabels([anion_label_map.get(g, g) for g in pivot.columns], rotation=0,
                       fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['tick_label'])

    # Simplify R-group labels for y-axis (matching Fig1.3 style)
    r_group_short_labels = {
        'Methyl': 'Me',
        'Ethyl': 'Et',
        'Propyl': 'n-Pr',
        'Isopropyl': 'i-Pr',
        '5-Member Ring': '5-Ring',
        '6-Member Ring': '6-Ring',
        '7-Member Ring': '7-Ring',
        'Oxy-6-Member': 'Oxy-6'
    }
    ax.set_yticklabels([r_group_short_labels.get(r, r) for r in pivot.index],
                       rotation=0, fontfamily='Times New Roman',
                       fontsize=PUBLICATION_FONTS['tick_label'])

    format_axis_labels(ax,
                     xlabel='Anion Functional Group',
                     ylabel='R Group',
                     title=None)

    plt.tight_layout()
    return fig, ax


def create_deltaG_distribution_six_subgroups(df):
    """Create manuscript Figure 2a: ΔG_rel boxplot over six terminal subgroups.

    Style mirrors generate_patent_core_figures.plot_deltag_distribution (the
    shared 5-group version with A_N merged), but splits A_N into its
    OH-terminated (hA) and O⁻-terminated (hN) subgroups so the x-axis aligns
    with the Figure 2b heatmap and Figure 3. The shared common_figures
    figure is not touched: callers save the result into manuscript/Figures.

    Fixed-layout version (2026-09-04): the axes box is placed with add_axes on
    a fixed 7.8 x 5.975 in canvas and saved without bbox_inches='tight', so
    the PNG has exact pixel dimensions and assembles 1:1 into the Figure 2
    grid. The previous tight-bbox output sizes forced different per-tile
    rescale factors in assemble_fig2, which shrank the A/B fonts relative to
    C/D. The axes geometry is shared with Figure 2b/2c/2d: axes 6.30 x 4.725
    in (exactly 8 x 6 square heatmap cells), left margin 1.35 in, bottom
    1.10 in, top 0.15 in.
    """
    plt.rcParams.update(get_publication_style())
    fig = plt.figure(figsize=(7.8, 5.975), dpi=300)
    ax = fig.add_axes([1.35 / 7.8, 1.10 / 5.975, 6.30 / 7.8, 4.725 / 5.975])

    def extract_subgroup(name):
        prefix = str(name)[1].upper()
        return prefix if prefix in SUBGROUP_COLORS else 'Unknown'

    subgroups = df['Name'].apply(extract_subgroup)
    data_by_subgroup = [
        df.loc[subgroups == subgroup, 'dG_rel_kcal_per_mol'].to_numpy()
        for subgroup in SUBGROUP_ORDER
    ]
    for subgroup, values in zip(SUBGROUP_ORDER, data_by_subgroup):
        print(f"  {subgroup}: n = {len(values)}")

    positions = np.arange(1, len(SUBGROUP_ORDER) + 1)
    bp = ax.boxplot(
        data_by_subgroup,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=False
    )

    for patch, subgroup in zip(bp['boxes'], SUBGROUP_ORDER):
        patch.set_facecolor(SUBGROUP_COLORS[subgroup])
        patch.set_edgecolor('black')
        # Same box fill strength as Fig. 2d so the two panels' colours match.
        patch.set_alpha(0.6)

    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.6)

    rng = np.random.default_rng(42)
    for idx, (values, subgroup) in enumerate(zip(data_by_subgroup, SUBGROUP_ORDER), 1):
        ax.scatter(idx, values.mean(), marker='D', color='darkred', s=60, zorder=5)

        quartiles = np.percentile(values, [25, 50, 75])
        ax.hlines(quartiles[0], idx - 0.2, idx + 0.2, colors='black', linestyles='--', linewidth=1.2)
        ax.hlines(quartiles[1], idx - 0.25, idx + 0.25, colors='black', linestyles='-', linewidth=1.5)
        ax.hlines(quartiles[2], idx - 0.2, idx + 0.2, colors='black', linestyles='--', linewidth=1.2)

        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(np.full_like(values, idx) + jitter, values,
                   color=SUBGROUP_COLORS[subgroup], edgecolors='black',
                   alpha=0.45, s=35, linewidths=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels([SUBGROUP_LABELS[s] for s in SUBGROUP_ORDER],
                       fontsize=PUBLICATION_FONTS['tick_label'],
                       fontfamily='Times New Roman')
    ax.set_xlabel('Terminal Chemistry', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=14)
    ax.set_ylabel(r'$\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                  fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    ax.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')

    return fig, ax


def create_deltaG_heatmap_six_subgroup_formulas(df):
    """Create manuscript Figure 2b: 6 terminal-subgroup x 8 R-group heatmap.

    Transposed layout (2026-09-03 user request): 6 rows of terminal chemistry
    (A/N/C/H/P/S) x 8 columns of R groups (Me/Et/n-Pr/i-Pr/5-Ring/6-Ring/
    7-Ring/Oxy-6), so the panel is a wide rectangle matching Figure 2a in the
    2x2 Figure 2 assembly. The shared common_figures heatmap
    (create_highthroughput_heatmap) keeps its original 8-row orientation and
    is not touched.

    Fixed-layout version (2026-09-04): axes and colourbar are placed with
    add_axes on a fixed 8.95 x 7.425 in canvas (same axes box and left margin
    as Figure 2a, plus 1.30 in on the right for the colourbar block) and the
    figure is saved without bbox_inches='tight', so the PNG assembles 1:1
    into the Figure 2 grid with no font rescaling. The 2.55 in bottom margin
    is sized to hold tick pad + 30-degree rotated labels + the x label;
    with the earlier 1.55 in margin the "R Group" label was clipped off the
    canvas edge. Note matplotlib measures the xlabel labelpad from the tick
    label bounding box (not the spine), so a small pad already clears the
    rotated labels. The colourbar uses an
    explicit FixedLocator/FixedFormatter so every tick label (including
    >= 3.0) is created once with Times New Roman — the previous auto-locator
    recreated the upper labels at draw time with the rcParams sans-serif
    fallback. R-group tick labels go through center_rotated_xticklabels so
    each label's geometric centre sits exactly on its column centre.
    """
    import matplotlib.ticker as mticker

    plt.rcParams.update(get_publication_style())
    # Canvas width 9.60 in: the colourbar label is anchored past the tick
    # labels (labelpad is measured from the Axes bounding edge incl. ticks),
    # so the canvas must extend right of the colourbar or the label is
    # clipped; 0.27 in of white is left of the label after it.
    fig = plt.figure(figsize=(9.60, 7.425), dpi=300)
    ax = fig.add_axes([1.35 / 9.60, 2.55 / 7.425, 6.30 / 9.60, 4.725 / 7.425])
    cax = fig.add_axes([(1.35 + 6.30 + 0.26) / 9.60, 2.55 / 7.425,
                        0.20 / 9.60, 4.725 / 7.425])

    data = df.copy()
    data['r_group'] = data['Name'].apply(classify_r_group)
    data['subgroup'] = data['Name'].apply(
        lambda name: str(name)[1].upper() if str(name)[1].upper() in SUBGROUP_COLORS
        else 'Unknown')

    pivot = data.pivot_table(values='dG_rel_kcal_per_mol', index='subgroup',
                             columns='r_group', aggfunc='mean')
    pivot = pivot.reindex(index=SUBGROUP_ORDER, columns=R_GROUP_ORDER)

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                center=df['dG_rel_kcal_per_mol'].mean(),
                annot_kws={'fontsize': 26, 'fontfamily': 'Times New Roman'},
                square=True, cbar=False, ax=ax)

    # Dedicated colourbar axes with fixed ticks: every label is created once
    # with explicit properties (fixes the >=3.0 sans-serif fallback).
    cbar = fig.colorbar(ax.collections[0], cax=cax)
    ticks = np.arange(0.5, 4.01, 0.5)
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(
        mticker.FixedFormatter([f'{t:.1f}' for t in ticks]))
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(PUBLICATION_FONTS['colorbar'])
    cbar.ax.tick_params(axis='y', length=4, pad=6)
    cbar.ax.set_ylabel(r'Mean $\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                       fontsize=PUBLICATION_FONTS['colorbar'] - 2, fontweight='bold',
                       fontfamily='Times New Roman', labelpad=8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('black')

    ax.set_yticklabels([SUBGROUP_LABELS.get(s, s) for s in pivot.index],
                       rotation=0, fontfamily='Times New Roman',
                       fontsize=PUBLICATION_FONTS['tick_label'])
    ax.tick_params(axis='y', pad=8)

    # Heatmap cell centres sit at data coordinates 0.5, 1.5, ..., 7.5.
    center_rotated_xticklabels(
        ax,
        [R_GROUP_SHORT_LABELS[r] for r in pivot.columns],
        ticks=[i + 0.5 for i in range(len(pivot.columns))],
        rotation=30,
        pad=9,
    )
    ax.set_xlabel('R Group', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=14)
    ax.set_ylabel('Terminal Chemistry', fontsize=PUBLICATION_FONTS['axis_label'],
                  fontfamily='Times New Roman', fontweight='bold', labelpad=14)

    return fig, ax


def create_supplementary_descriptor_scatters(df):
    """Create manuscript Supplementary Fig. S4: descriptor vs ΔG_rel scatters.

    Three panels (Molecular Weight, Max Partial Charge, Labute ASA against
    ΔG_rel), using the same descriptor values as every other figure via
    compute_structure_descriptors, and the same six-subgroup coloring as
    Figure 2a. The shared common_figures combined figure is untouched.
    """
    from rdkit import Chem  # noqa: F401  (keeps rdkit import explicit like the other paths)

    df = df.copy()
    desc = df['SMILES'].apply(compute_structure_descriptors)
    df[['mol_wt', 'logP', 'ASA', 'max_charge', 'n_rotatable']] = pd.DataFrame(
        desc.tolist(), index=df.index)

    def extract_subgroup(name):
        prefix = str(name)[1].upper()
        return prefix if prefix in SUBGROUP_COLORS else 'Unknown'

    df['subgroup'] = df['Name'].apply(extract_subgroup)

    panels = [
        ('mol_wt', 'Molecular Weight (Da)'),
        ('max_charge', 'Max Partial Charge (|e|)'),
        ('ASA', r'Labute ASA (Å$^2$)'),
    ]

    # The figure-level legend is anchored at the canvas bottom edge
    # (loc='upper center', y=0.0): it can then never collide with the x-axis
    # labels, and bbox_inches='tight' in save_publication_figure extends the
    # saved area to include it. Reserving space via tight_layout(rect=...) or
    # constrained layout does not survive save_publication_figure's second
    # fig.tight_layout() call, which previously caused the overlap.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (column, xlabel) in zip(axes, panels):
        for subgroup in SUBGROUP_ORDER:
            part = df[df['subgroup'] == subgroup]
            ax.scatter(part[column], part['dG_rel_kcal_per_mol'],
                       c=SUBGROUP_COLORS[subgroup], alpha=0.75, s=40,
                       edgecolors='none', label=SUBGROUP_LABELS[subgroup])
        ax.set_xlabel(xlabel, fontsize=PUBLICATION_FONTS['axis_label'],
                      fontfamily='Times New Roman', fontweight='bold')
        ax.set_ylabel(r'$\Delta G_{\mathrm{rel}}$ ($\mathrm{kcal\cdot mol^{-1}}$)',
                      fontsize=PUBLICATION_FONTS['axis_label'],
                      fontfamily='Times New Roman', fontweight='bold')
        ax.tick_params(axis='both', labelsize=PUBLICATION_FONTS['tick_label'])
        ax.grid(False)
        ax.set_axisbelow(True)

    handles = [Line2D([0], [0], marker='o', color='none',
                      markerfacecolor=SUBGROUP_COLORS[s], markersize=9,
                      markeredgecolor='none', label=SUBGROUP_LABELS[s])
               for s in SUBGROUP_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False,
               prop={'family': 'Times New Roman', 'size': PUBLICATION_FONTS['legend']},
               bbox_to_anchor=(0.5, 0.06))

    return fig, axes


def main():
    """Main function to generate all data analysis figures."""
    parser = argparse.ArgumentParser(description='Generate OP_unit data analysis figures')
    parser.add_argument(
        '--pmf-selection', choices=['original', 'fixed-rgroup'], default=None,
        help='Regenerate only the Figure 3 PMF figure with the given panel '
             'selection (original = historical selection; fixed-rgroup = the '
             'same three R-groups in every class). Output goes to '
             'manuscript/Figures so the shared common_figures are untouched.',
    )
    parser.add_argument(
        '--s6', action='store_true',
        help='Generate Supplementary Fig. S2 (additional PMF profiles) into '
             'manuscript/Figures/.',
    )
    parser.add_argument(
        '--legacy-pmf-colors', action='store_true',
        help='Keep the historical Dark2 curve colours and h-prefixed motif '
             'identifiers in the PMF legends instead of the manuscript '
             'eight-colour R-group palette (2026-09-03). Only affects the '
             '--pmf-selection / --s6 manuscript modes.',
    )
    parser.add_argument(
        '--fig2a', action='store_true',
        help='Generate manuscript Figure 2a (ΔG_rel boxplot over the six '
             'terminal subgroups, A_N split into OH/O⁻) into '
             'manuscript/Figures/.',
    )
    parser.add_argument(
        '--fig2c', action='store_true',
        help='Generate manuscript Figure 2c (8 R-groups x 6 terminal-subgroup '
             'mean ΔG_rel heatmap with chemical-formula labels) into '
             'manuscript/Figures/.',
    )
    parser.add_argument(
        '--s2', action='store_true',
        help='Generate Supplementary Fig. S4 (MW / max partial charge / '
             'Labute ASA vs ΔG_rel, six-subgroup coloring) into '
             'manuscript/Figures/.',
    )
    args = parser.parse_args()

    # PMF-only mode: never write to the shared common_figures directory.
    if args.pmf_selection is not None or args.s6 or args.fig2a or args.fig2c or args.s2:
        MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        if args.pmf_selection is not None:
            print(f"Creating PMF curves figure (selection={args.pmf_selection})...")
            try:
                fig, _ax = create_pmf_curves_figure(
                    selection=args.pmf_selection, class_code_titles=True,
                    rgroup_colors=not args.legacy_pmf_colors,
                    fill_alpha=MANUSCRIPT_PMF_FILL_ALPHA,
                    align_replicates=True, pmf_ylabel=True)
                out_name = (
                    'Fig3_PMF_Curves_original' if args.pmf_selection == 'original'
                    else 'Fig3_PMF_Curves'
                )
                save_path = MANUSCRIPT_FIGURE_DIR / f'{out_name}.png'
                if args.pmf_selection == 'fixed-rgroup':
                    # Structure-strip figure sets its own gridspec geometry;
                    # save directly (as with the Fig. 2 tiles) so the explicit
                    # layout survives — save_publication_figure would re-flow
                    # it with tight_layout.
                    fig.savefig(str(save_path), dpi=300, bbox_inches=None,
                                facecolor='white', edgecolor='none')
                else:
                    save_publication_figure(fig, str(save_path))
                print(f"  saved: {save_path}")
            except Exception as e:
                print(f"Error creating PMF curves figure: {e}")
        if args.s6:
            print("Creating supplementary PMF figure (Supplementary Fig. S5)...")
            try:
                fig, _ax = create_supplementary_pmf_figure(
                    class_code_titles=True,
                    rgroup_colors=not args.legacy_pmf_colors,
                    fill_alpha=MANUSCRIPT_PMF_FILL_ALPHA,
                    align_replicates=True, pmf_ylabel=True)
                save_path = MANUSCRIPT_FIGURE_DIR / 'FigS5_Additional_PMF_Profiles.png'
                save_publication_figure(fig, str(save_path))
                print(f"  saved: {save_path}")
            except Exception as e:
                print(f"Error creating supplementary PMF figure: {e}")
        if args.fig2a:
            print("Creating manuscript Figure 2a (six-subgroup ΔG_rel boxplot)...")
            try:
                df = load_molecular_data()
                fig, _ax = create_deltaG_distribution_six_subgroups(df)
                save_path = MANUSCRIPT_FIGURE_DIR / 'Fig2a_DeltaG_Distribution_6subgroups.png'
                # Fixed-layout tile: save at exact pixel size for 1:1 assembly.
                fig.savefig(str(save_path), dpi=300, bbox_inches=None,
                            facecolor='white', edgecolor='none')
                print(f"  saved: {save_path}")
            except Exception as e:
                print(f"Error creating six-subgroup distribution figure: {e}")
        if args.fig2c:
            print("Creating manuscript Figure 2c (8x6 mean ΔG_rel heatmap)...")
            try:
                df = load_molecular_data()
                fig, _ax = create_deltaG_heatmap_six_subgroup_formulas(df)
                save_path = MANUSCRIPT_FIGURE_DIR / 'Fig2c_Mean_dGrel_Heatmap_8x6.png'
                # Fixed-layout tile: save at exact pixel size for 1:1 assembly.
                fig.savefig(str(save_path), dpi=300, bbox_inches=None,
                            facecolor='white', edgecolor='none')
                print(f"  saved: {save_path}")
            except Exception as e:
                print(f"Error creating manuscript heatmap figure: {e}")
        if args.s2:
            print("Creating supplementary descriptor scatters (Supplementary Fig. S4)...")
            try:
                df = load_molecular_data()
                fig, _axes = create_supplementary_descriptor_scatters(df)
                save_path = MANUSCRIPT_FIGURE_DIR / 'FigS4_Descriptor_Scatters_6subgroups.png'
                save_publication_figure(fig, str(save_path))
                print(f"  saved: {save_path}")
            except Exception as e:
                print(f"Error creating supplementary descriptor scatters: {e}")
        return

    print("Generating data analysis figures...")

    # Load molecular data
    try:
        df = load_molecular_data()
        print(f"Successfully loaded {len(df)} molecules")
    except Exception as e:
        print(f"Error loading molecular data: {e}")
        return

    figures = []

    # Generate combined data analysis figure (replaces Fig1_3 and Fig1_4)
    print("\n1. Creating combined data analysis figure (Fig 1.3)...")
    try:
        fig, _axes = create_combined_data_figure(df)
        save_figure(fig, 'Fig1_3_Combined_Data_Analysis')
        figures.append('Fig1_3_Combined_Data_Analysis')
    except Exception as e:
        print(f"Error creating combined data analysis figure: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. Creating PMF curves figure...")
    try:
        fig, ax = create_pmf_curves_figure()
        save_figure(fig, 'Fig1_4_PMF_Curves')
        figures.append('Fig1_4_PMF_Curves')
    except Exception as e:
        print(f"Error creating PMF curves figure: {e}")

    # Prediction scatter using retrained models
    try:
        print("\n4. Loading retrained predictions...")
        predictions_df = load_retrained_predictions('improved')
        if predictions_df.empty:
            raise RuntimeError("No retrained predictions found")

        print("\n5. Creating prediction scatter plot...")
        fig, ax = create_prediction_scatter_figure(predictions_df)
        save_figure(fig, 'Fig1_14_Prediction_Scatter')
        figures.append('Fig1_14_Prediction_Scatter')

    except Exception as e:
        print(f"Could not create prediction scatter plot: {e}")

    print("\n6. Creating high-throughput prediction heatmap...")
    try:
        fig, _ax = create_highthroughput_heatmap(df)
        save_figure(fig, 'Fig1_11_Highthroughput_Prediction')
        figures.append('Fig1_11_Highthroughput_Prediction')
    except Exception as e:
        print(f"Error creating high-throughput heatmap: {e}")

    print(f"\n✅ Generated {len(figures)} data analysis figures:")
    for fig_name in figures:
        print(f"   - {fig_name}")

if __name__ == "__main__":
    main()
