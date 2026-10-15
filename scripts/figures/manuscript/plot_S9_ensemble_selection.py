#!/usr/bin/env python3
"""
Generate Supplementary Fig. S9: Ensemble selection and model diversity.

Layout: Horizontal (2 panels side by side)
  a. Best OOF MAE versus ensemble size (K=1..5, line plot with markers)
  b. Base models versus ensemble (5 base model MAEs + ensemble MAE, bar + horizontal line)

Data source: data/ensemble_oof.csv, which contains the tagged best-per-K,
per-member and fold-metric sections used below.

Output: manuscript/Figures/FigS9_Ensemble_Selection.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path setup
BASE = Path(__file__).resolve().parents[3]  # figures/manuscript/ -> scripts/figures -> scripts -> OP_unit
sys.path.insert(0, str(BASE / "scripts" / "figures"))

from plot_utils import (
    PUBLICATION_FONTS, ANION_COLORS, STRICT_FILL, apply_publication_style,
    save_publication_figure
)
from public_data import load_ensemble_oof


def panel_a_ensemble_size_sweep(ax):
    """Panel a: Best OOF MAE versus ensemble size (K=1..5)."""
    per_model = load_ensemble_oof("oof_per_model")
    bpk = load_ensemble_oof("best_per_k").sort_values("n_members")

    # K=1 = best single model OOF MAE; K=2..5 = best-per-K from exhaustive search
    k1_mae = float(per_model["oof_mae"].min())
    k_values = [1] + bpk["n_members"].astype(int).tolist()
    mae_values = [k1_mae] + bpk["oof_mae"].tolist()

    ax.plot(k_values, mae_values, 'o-', color='#756bb1', linewidth=2.5, markersize=8)

    ax.set_xlabel('Ensemble size (K)', fontfamily='Times New Roman',
                  fontsize=PUBLICATION_FONTS['axis_label'])
    ax.set_ylabel('Best OOF MAE ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim([0.265, 0.32])

    # 标注每个 K 的 MAE，直观展示增益饱和
    for k, m in zip(k_values, mae_values):
        ax.annotate(f'{m:.3f}', (k, m), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontfamily='Times New Roman',
                    fontsize=PUBLICATION_FONTS['annotation'])

    ax.text(-0.18, 1.15, 'a', transform=ax.transAxes, fontsize=40,
            fontweight='bold', va='top', ha='left', fontfamily='Times New Roman')


def panel_b_base_vs_ensemble(ax):
    """Panel b: Base models versus ensemble (data-driven)."""
    df = load_ensemble_oof("oof_per_model")

    # 选中 ensemble 的 5 个成员，按 OOF MAE 升序（与图注/Table S5 顺序一致）
    member_keys = [
        ("svm_rbf", "morgan_feature_hashed_count_fp_r2_8192"),
        ("mlp", "deepchem_mol2vec"),
        ("svm_rbf", "cdk_lingo"),
        ("xgboost", "datamol_secfp"),
        ("random_forest", "morgan_feature_hashed_count_fp_r2_8192"),
    ]
    base_names = ['SVM-RBF\nMorgan', 'MLP\nMol2Vec', 'SVM-RBF\nLingo', 'XGBoost\nSECFP', 'RF\nMorgan']
    base_mae = []
    for mod, rep in member_keys:
        row = df[(df["model"] == mod) & (df["rep"] == rep)]
        if len(row) != 1:
            raise KeyError(f"Per-model OOF MAE not found for {mod}|{rep}")
        base_mae.append(float(row["oof_mae"].iloc[0]))

    # 冻结的最终 ensemble OOF MAE（5 折均值 = pooled，5 折样本数相等）
    ensemble_mae = float(load_ensemble_oof("fold_metrics")["mae"].mean())

    x = np.arange(len(base_names))
    ax.bar(x, base_mae, width=0.6, color=STRICT_FILL, edgecolor='black', linewidth=1.2)

    # Ensemble 红线 + 内联标注（不用 legend）。文字右对齐收在轴内、线右端
    # 上方，白底衬垫避免与柱体填充混淆；不再像旧版那样把文字挤到轴外。
    ax.axhline(ensemble_mae, color=ANION_COLORS['P'], linestyle='solid', linewidth=2.5)
    ax.text(x[-1] + 0.45, ensemble_mae + 0.004, f'Ensemble = {ensemble_mae:.3f}',
            fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['annotation'],
            ha='right', va='bottom', color=ANION_COLORS['P'],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

    ax.set_xticks(x)
    # 两行标签最长 7 字符（SVM-RBF/XGBoost/Mol2Vec）：26pt 横排在 ~90pt 的
    # 刻度间距里必然互相擦碰（旋转也只能错开方向、放不下整个标签块），
    # 按"长标签缩短/调字号"规范在该面板把 tick 降到 20pt 保持水平排布
    ax.set_xticklabels(base_names,
                       fontfamily='Times New Roman', fontsize=20)
    ax.set_ylabel('OOF MAE ($\\mathrm{kcal\\cdot mol^{-1}}$)',
                  fontfamily='Times New Roman', fontsize=PUBLICATION_FONTS['axis_label'])
    ax.set_ylim([0.25, 0.40])
    ax.set_xlim(-0.6, x[-1] + 0.7)

    ax.text(-0.10, 1.15, 'b', transform=ax.transAxes, fontsize=40,
            fontweight='bold', va='top', ha='left', fontfamily='Times New Roman')


def main():
    apply_publication_style()

    # Horizontal layout: 2 panels side by side (match other SI figures' scale)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8))

    panel_a_ensemble_size_sweep(ax_a)
    panel_b_base_vs_ensemble(ax_b)

    plt.tight_layout()

    out_path = BASE / "manuscript" / "Figures" / "FigS9_Ensemble_Selection.png"
    save_publication_figure(fig, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
