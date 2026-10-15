#!/usr/bin/env python3
"""Generate Supplementary Fig. S10: Applicability domain and inter-model disagreement.

Layout: Horizontal (2 panels side by side)
  a. Chemical space distribution (PCA of 120 training + prospective candidates, Morgan FP)
  b. Inter-model SD vs absolute prospective prediction error (default)

By default panel a shows the full 32-candidate prospective library: the 24
non-selected candidates in grey and the 8 PMF-selected amine-oxide motifs
(N035, N055, N067, N103, N123, N127, N143, N159) highlighted as orange
triangles, alongside the 120 training motifs; panel b shows the
SD-vs-absolute-error view for the 8 evaluated motifs (the only ones with PMF
values). This matches the 32→8 prospective narrative in manuscript_v1.md.
Pass ``--all-32`` to reproduce the earlier 32-candidate figure for
record-keeping (single candidate group in panel a, with the legacy panel b,
ensemble mean vs inter-model SD); pass ``--legacy-sd-plot`` to reproduce the
earlier eight-candidate view with the legacy ensemble-mean-vs-SD panel b.

Output: manuscript/Figures/FigS10_Applicability_Domain.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.decomposition import PCA

# Path setup
BASE = Path(__file__).resolve().parents[3]  # figures/manuscript/ -> scripts/figures -> scripts -> OP_unit
sys.path.insert(0, str(BASE / "scripts" / "figures"))

from plot_utils import (  # noqa: E402
    PUBLICATION_FONTS,
    ANION_COLORS,
    apply_publication_style,
    save_publication_figure,
)
from public_data import load_prospective_library, load_prospective_pmf, load_reference_table

# Eight motifs carried forward to prospective PMF evaluation (Table S6).
PROSPECTIVE_NAMES = [
    "N035", "N055", "N067", "N103", "N123", "N127", "N143", "N159",
]

def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute a Morgan fingerprint as a numpy bit array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    return np.array(fp)


def load_candidates(use_all_32: bool) -> pd.DataFrame:
    """Return the candidate prediction rows for the figure.

    Args:
        use_all_32: if True, keep all 32 candidate rows (legacy record-keeping
            view); if False, keep only the eight prospectively evaluated motifs.
    """
    df = load_prospective_library()
    if use_all_32:
        return df
    return df[df["Name"].isin(PROSPECTIVE_NAMES)].copy()


def merge_pmf_values(df_cand: pd.DataFrame) -> pd.DataFrame:
    """Join PMF-derived deltaG_rel onto the candidate predictions.

    Uses the same PMF summary and hN->N renaming convention as
    ``plot_fig6_prospective.py``; the signed error is predicted minus PMF.
    """
    pmf = load_prospective_pmf()[["Name", "PMF_dG_rel"]].rename(
        columns={"PMF_dG_rel": "pmf_dG"}
    )
    df = df_cand.merge(pmf, on="Name", validate="1:1")
    df["abs_error"] = (df["predicted_dG_rel"] - df["pmf_dG"]).abs()
    return df


def panel_a_chemical_space_pca(
    ax, df_cand: pd.DataFrame, highlight_selected: bool = False
) -> None:
    """Panel a: PCA of 120 training motifs and the prospective candidates.

    Args:
        ax: matplotlib axes to draw on.
        df_cand: candidate prediction rows to show.
        highlight_selected: if True, split the candidates into the 24
            non-selected motifs (grey circles) and the 8 PMF-selected motifs
            (orange triangles, highlighted) instead of plotting all candidates
            in one orange group.
    """
    # Load training set (120 motifs)
    df_train = load_reference_table()
    train_smiles = df_train["SMILES"].values
    cand_smiles = df_cand["SMILES"].values

    train_fps = np.array([compute_morgan_fp(s) for s in train_smiles])
    cand_fps = np.array([compute_morgan_fp(s) for s in cand_smiles])

    # Concatenate and run PCA
    all_fps = np.vstack([train_fps, cand_fps])
    pca = PCA(n_components=2, random_state=42)
    all_pca = pca.fit_transform(all_fps)

    train_pca = all_pca[: len(train_fps)]
    cand_pca = all_pca[len(train_fps):]

    # Plot training set (blue)
    ax.scatter(
        train_pca[:, 0], train_pca[:, 1], s=30, alpha=0.5, c=ANION_COLORS["A_N"],
        edgecolors="none", label="Training (120)",
    )

    if highlight_selected:
        is_selected = df_cand["Name"].isin(PROSPECTIVE_NAMES).values
        # Non-selected candidates (grey, small circles, drawn first so the
        # highlighted triangles sit on top).
        ax.scatter(
            cand_pca[~is_selected, 0], cand_pca[~is_selected, 1],
            s=40, alpha=0.55, c="#bbbbbb", edgecolors="black", linewidths=0.5,
            label=f"Predicted, not selected ({int((~is_selected).sum())})",
        )
        # PMF-selected candidates (orange triangles).
        ax.scatter(
            cand_pca[is_selected, 0], cand_pca[is_selected, 1],
            s=70, alpha=0.85, c="#fd8d3c", edgecolors="black", linewidths=1.0,
            marker="^", label=f"Selected for PMF ({int(is_selected.sum())})",
        )
    else:
        n_cand = len(cand_fps)
        ax.scatter(
            cand_pca[:, 0], cand_pca[:, 1], s=60, alpha=0.8, c="#fd8d3c",
            edgecolors="black", linewidths=1.0,
            label=f"Candidates ({n_cand})", marker="^",
        )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )
    ax.legend(
        fontsize=PUBLICATION_FONTS["legend"], loc="best", frameon=True,
        prop={"family": "Times New Roman"},
    )

    ax.text(
        -0.15, 1.08, "a", transform=ax.transAxes,
        fontsize=PUBLICATION_FONTS["title"], fontweight="bold",
        va="top", ha="left", fontfamily="Times New Roman",
    )


def panel_b_mean_vs_sd(ax, df_cand: pd.DataFrame) -> None:
    """Panel b (legacy): Ensemble mean vs inter-model SD scatter.

    Retained so the earlier published version can be reproduced via
    ``--legacy-sd-plot`` or ``--all-32``.
    """
    x = df_cand["predicted_dG_rel"].values
    y = df_cand["ensemble_sd"].values
    candidate_ids = df_cand["Name"].values

    ax.scatter(
        x, y, s=50, alpha=0.7, c="#fd8d3c",
        edgecolors="black", linewidths=0.8,
    )

    # Highlight and annotate the candidate with the highest inter-model SD.
    max_sd_idx = int(np.argmax(y))
    max_sd_name = candidate_ids[max_sd_idx]
    ax.scatter(
        x[max_sd_idx], y[max_sd_idx], s=150, facecolors="none", edgecolors="red",
        linewidths=2.5, label=f"{max_sd_name} (highest SD)",
    )
    ax.legend(
        fontsize=PUBLICATION_FONTS["legend"], loc="upper right", frameon=True,
        prop={"family": "Times New Roman"},
    )

    ax.set_xlabel(
        r"Ensemble mean $\Delta G_\mathrm{rel}$ ($\mathrm{kcal\cdot mol^{-1}}$)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )
    ax.set_ylabel(
        rf"Inter-model SD ($\mathrm{{kcal\cdot mol^{{-1}}}}$)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )

    ax.text(
        -0.15, 1.08, "b", transform=ax.transAxes,
        fontsize=PUBLICATION_FONTS["title"], fontweight="bold",
        va="top", ha="left", fontfamily="Times New Roman",
    )


def panel_b_sd_vs_abs_error(ax, df_cand: pd.DataFrame) -> None:
    """Panel b: Inter-model SD vs absolute prospective prediction error.

    The prospective PMF results are joined from ``data/prospective_pmf_8.csv``; the
    absolute error is |ensemble mean - PMF-derived deltaG_rel|. This plot shows
    the exploratory association between ensemble disagreement and prediction
    error (Spearman rho ~ 0.90 for the eight candidates), which replaced the
    earlier ensemble-mean-vs-SD panel because the mean-vs-SD association was
    weak and negative once computed on the eight prospective motifs.
    """
    df = merge_pmf_values(df_cand)
    x = df["ensemble_sd"].values
    y = df["abs_error"].values
    candidate_ids = df["Name"].values

    ax.scatter(
        x, y, s=50, alpha=0.7, c="#fd8d3c",
        edgecolors="black", linewidths=0.8, zorder=3,
    )

    # Label every candidate. Offsets are chosen per point so the label is always
    # pushed toward the interior of the axes (away from the nearest edge), which
    # keeps even the top-right outlier (N159) inside the frame and avoids overlap.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
    # Manual label offsets (points) for points whose automatic placement would
    # collide. N123 (leftmost) and N067 sit very close together; push N123's
    # label straight up and N067's straight right so they stay apart and clear of
    # the axis edges.
    label_offset = {"N123": (0, 16), "N067": (24, 0)}
    for xi, yi, name in zip(x, y, candidate_ids):
        if name in label_offset:
            dx, dy = label_offset[name]
        else:
            dx = -18 if xi > xmid else 14
            dy = -18 if yi > ymid else 14
        ax.annotate(
            name, (xi, yi), textcoords="offset points", xytext=(dx, dy),
            ha="center", va="center",
            fontsize=PUBLICATION_FONTS["tick_label"],
            fontfamily="Times New Roman",
        )

    r_pearson = pearsonr(x, y).statistic
    r_spearman = spearmanr(x, y).statistic

    # Ordinary least-squares fit of absolute error on inter-model SD, drawn so
    # the association the statistics summarise is visible. The fit line is the
    # legend handle and its label carries the Pearson/Spearman coefficients
    # (no group key is needed here, since only the eight candidates are shown).
    slope, intercept, _, _, _ = linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    # OLS best-fit reference line. Colour changed from black to a top-journal
    # palette blue (#0072B2, Okabe-Ito, colour-blind safe and the standard
    # Nature/Science accent) so it reads clearly against the orange markers;
    # dashed, linewidth 2, alpha 0.85, drawn behind the markers (no label, so
    # it is not echoed in the legend).
    ax.plot(
        xs, slope * xs + intercept,
        "--", color="#756bb1", linewidth=2, alpha=0.85, zorder=1,
    )
    # Legend carries only the Pearson/Spearman coefficients as pure text: an
    # empty Line2D handle with handlelength 0 means no line and no marker handle,
    # so the box reads as text only while keeping the 10a-style frame/shadow/font.
    ax.legend(
        [plt.Line2D([], [], linestyle="none", marker="")],
        [f"Pearson $r$ = {r_pearson:.2f}\nSpearman $\\rho$ = {r_spearman:.2f}"],
        loc="lower right", frameon=True,
        fontsize=PUBLICATION_FONTS["legend"],
        prop={"family": "Times New Roman"},
        handlelength=0, handletextpad=0.2,
    )

    ax.set_xlabel(
        rf"Inter-model SD ($\mathrm{{kcal\cdot mol^{{-1}}}}$)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )
    ax.set_ylabel(
        rf"Absolute error ($\mathrm{{kcal\cdot mol^{{-1}}}}$)",
        fontfamily="Times New Roman", fontsize=PUBLICATION_FONTS["axis_label"],
    )

    ax.text(
        -0.15, 1.08, "b", transform=ax.transAxes,
        fontsize=PUBLICATION_FONTS["title"], fontweight="bold",
        va="top", ha="left", fontfamily="Times New Roman",
    )


def main() -> None:
    """Build and save Supplementary Fig. S10."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-32", action="store_true",
        help="Reproduce the legacy 32-candidate figure (record-keeping only): "
             "all 32 candidates as one orange group in panel a, legacy "
             "ensemble-mean-vs-SD panel b.",
    )
    parser.add_argument(
        "--legacy-sd-plot", action="store_true",
        help="Use the legacy panel b (ensemble mean vs inter-model SD) for the "
             "eight-candidate view, written to a separate _sd filename; the "
             "default panel b is inter-model SD vs absolute prediction error.",
    )
    args = parser.parse_args()

    apply_publication_style()

    # Horizontal layout: 2 panels side by side (match other SI figures' scale)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))

    df_cand = load_candidates(use_all_32=not args.legacy_sd_plot)
    panel_a_chemical_space_pca(
        ax_a, df_cand,
        highlight_selected=not (args.all_32 or args.legacy_sd_plot),
    )
    if args.all_32 or args.legacy_sd_plot:
        panel_b_mean_vs_sd(ax_b, df_cand)
    else:
        # Panel b needs PMF values, which exist only for the 8 selected motifs;
        # restrict to those rows even though panel a shows all 32 candidates.
        panel_b_sd_vs_abs_error(
            ax_b, df_cand[df_cand["Name"].isin(PROSPECTIVE_NAMES)].copy()
        )

    plt.tight_layout()

    # Default (120 + 32-candidate library, 8 highlighted) overwrites the
    # published path; the legacy views are written to separate filenames.
    if args.all_32:
        suffix = "_all32"
    elif args.legacy_sd_plot:
        suffix = "_sd"
    else:
        suffix = ""
    out_path = (
        BASE / "manuscript" / "Figures"
        / f"FigS10_Applicability_Domain{suffix}.png"
    )
    save_publication_figure(fig, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
