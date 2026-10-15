#!/usr/bin/env python3
"""从 PMF profile.xvg 提取 ΔG_rel，按分子聚合跨副本 mean±SD，输出 CSV。

ΔG_rel 定义：G(z=1.6 nm) − G(z=4.0 nm)，与 manuscript/plan_v0.md 一致，
是同一 umbrella sampling 协议下的相对膜亲和性指标。

输入：scripts/202609-all120-variousOP/pmf_profiles/<ligand>-<replica>/profile.xvg
输出：scripts/202609-all120-variousOP/pmf_summary.csv
      列：Name, dG_rel_kcal_per_mol, replicate_sd, n_replicas
      - Name: hN001 等（-1..-5 副本被聚合掉）
      - dG_rel_kcal_per_mol: 跨副本均值
      - replicate_sd: 跨副本样本标准差（5次独立repeat间的SD，ddof=1；非SEM）
      - n_replicas: 实际使用的副本数

若指定 --pred-csv，会把 ensemble_mean 之类的列 join 进来，并打印
与 PMF dG_rel_kcal_per_mol 对比的 MAE / RMSE / Pearson r。

注意 data/pmf_profiles/（120 主数据集 + 3-mer 的 1220 条平铺 xvg，Fig3/S1/S2 用）
与本脚本读取的 scripts/202609-all120-variousOP/pmf_profiles/（8 个新候选 × 5 副本的
目录结构，Fig 6 前瞻验证用）是两套不同的数据，只是目录同名。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPLOY_ROOT = Path(__file__).resolve().parents[1] / "202609-all120-variousOP"


Z_LB = 1.6  # nm, membrane-embedded reference
Z_RB = 4.0  # nm, bulk-water reference


def _parse_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 profile.xvg，返回 (z, U)。跳过 @/# 注释与 xmgrace 元数据。"""
    zs, us = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            zs.append(float(parts[0]))
            us.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(zs), np.asarray(us)


def _interpolate_u(zs: np.ndarray, us: np.ndarray, z_target: float) -> float:
    """线性插值取 U(z_target)。"""
    return float(np.interp(z_target, zs, us))


def extract_dg_ligand(replica_dirs: list[Path]) -> tuple[float, float, int]:
    """对一个分子的多个副本目录，返回 (deltaG_mean, sd_across_replicas, n_replicas)。"""
    per_replica = []
    for d in replica_dirs:
        xvg = d / "profile.xvg"
        if not xvg.exists():
            continue
        zs, us = _parse_profile(xvg)
        if zs.size < 2:
            continue
        # G(z) - G(z_ref) 归一化后差值就等于差值本身，直接相减即可
        per_replica.append(_interpolate_u(zs, us, Z_LB) - _interpolate_u(zs, us, Z_RB))
    if not per_replica:
        return float("nan"), float("nan"), 0
    arr = np.asarray(per_replica)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def collect_dirs(pmf_root: Path) -> dict[str, list[Path]]:
    """按分子聚合副本目录，支持每个分子出现不同数量的副本（有多少算多少）。"""
    groups: dict[str, list[Path]] = {}
    for d in sorted(pmf_root.iterdir()):
        if not d.is_dir():
            continue
        name, _, rep = d.name.rpartition("-")
        if not rep or not rep.isdigit():
            continue
        groups.setdefault(name, []).append(d)
    return groups


def compare_with_prediction(
    pmf_df: pd.DataFrame,
    pred_csv: Path,
    pred_col: str,
) -> None:
    """把 PMF dG_rel_kcal_per_mol 与 ensemble 预测列 join，打印 MAE/RMSE/Pearson r。"""
    pred = pd.read_csv(pred_csv)
    if pred_col not in pred.columns:
        raise ValueError(f"列 {pred_col!r} 不在 {pred_csv.name} 中；可用列：{list(pred.columns)}")
    if "Name" not in pred.columns:
        raise ValueError(f"{pred_csv.name} 缺少 Name 列")

    # pmf_df.Name 是 hN035，pred.Name 是 N035：统一前缀后再 join
    pmf_df = pmf_df.copy()
    pmf_df["Name_pred"] = pmf_df["Name"].str.replace(r"^hN", "N", regex=True)
    merged = pmf_df.merge(pred[["Name", pred_col]], left_on="Name_pred", right_on="Name", how="inner")
    merged = merged.dropna(subset=[pred_col, "dG_rel_kcal_per_mol"])

    if merged.empty:
        print(f"[extract_pmf_dG] 无匹配分子可对比 (pmf: {list(pmf_df['Name_pred'])} vs pred: {list(pred['Name'][:3])}...)")
        return

    y_true = merged["dG_rel_kcal_per_mol"].to_numpy()
    y_pred = merged[pred_col].to_numpy()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])

    print(f"\n[extract_pmf_dG] 对比 {pred_csv.name} :: {pred_col} vs PMF dG_rel_kcal_per_mol (n={len(merged)})")
    print(f"  MAE     = {mae:.4f} kcal/mol")
    print(f"  RMSE    = {rmse:.4f} kcal/mol")
    print(f"  Pearson r = {r:.4f}")
    merged_show = merged[["Name_pred", "dG_rel_kcal_per_mol", "replicate_sd", "n_replicas", pred_col]].copy()
    merged_show["signed_err"] = merged_show["dG_rel_kcal_per_mol"] - merged_show[pred_col]
    merged_show = merged_show.rename(columns={"dG_rel_kcal_per_mol": "pmf_dG", pred_col: "pred"})
    print(merged_show.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pmf-root",
        type=Path,
        default=DEPLOY_ROOT / "pmf_profiles",
        help="PMF 数据根目录，默认 scripts/202609-all120-variousOP/pmf_profiles/",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEPLOY_ROOT / "pmf_summary.csv",
        help="输出 CSV 路径",
    )
    ap.add_argument("--ligands", nargs="*", default=None, help="只处理指定的分子名（如 hN159 hN103）")
    ap.add_argument(
        "--pred-csv",
        type=Path,
        default=DEPLOY_ROOT / "run-9.3/deployment/prospective_32.csv",
        help="ML 预测 CSV，用于对比；默认 prospective_32.csv",
    )
    ap.add_argument(
        "--pred-col",
        default="ensemble_mean",
        help="预测列名，默认 ensemble_mean",
    )
    args = ap.parse_args()

    groups = collect_dirs(args.pmf_root)
    if args.ligands:
        groups = {k: v for k, v in groups.items() if k in args.ligands}

    rows = []
    for name in sorted(groups):
        mean, sd, n = extract_dg_ligand(groups[name])
        rows.append({"Name": name, "dG_rel_kcal_per_mol": mean, "replicate_sd": sd, "n_replicas": n})

    df = pd.DataFrame(rows).sort_values("Name").reset_index(drop=True)
    df.to_csv(args.output, index=False, float_format="%.6f")

    print(f"[extract_pmf_dG] {args.pmf_root} -> {args.output}")
    print(f"[extract_pmf_dG] ligands: {len(df)}, total replicas used: {df['n_replicas'].sum()}")
    print(df.to_string(index=False))

    if args.pred_csv.exists():
        compare_with_prediction(df, args.pred_csv, args.pred_col)
    else:
        print(f"[extract_pmf_dG] 未找到预测 CSV，跳过对比: {args.pred_csv}")


if __name__ == "__main__":
    main()
