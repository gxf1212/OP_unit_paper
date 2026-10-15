#!/usr/bin/env python3
"""Train best model(s) on all 120 OP molecules, then predict 32 new N-oxide candidates.

工作流（per plan-correction-9.2.md + 2026-09-03 ensemble 决策）:
  single 模式（原行为）:
    1. 从 Stage 1+2 筛出的 best Phase 1 candidate 取 top N
    2. 各用全 120 重训 + 预测 32，写 CSV + Stage3 deployment session（写回 DB）
  ensemble 模式:
    A. ensemble selection：候选池（HPO top N）用与 Stage 1 完全一致的固定 5-fold
       生成 OOF 预测，按 OOF MAE 做 greedy forward selection（等权平均），
       输出 ensemble_selection.csv / per-model OOF / residual corr
    B. final deployment：选中的 K 个组合各全 120 重训（HPO best params）+ 预测 32，
       输出每个 base model 预测 + ensemble mean ± SD（inter-model disagreement）+ rank
       —— 纯 CSV，不写回 screening.db（32 个无真值，避免污染 CV 指标）

用法:
    python retrain_best_model.py --mode single --top-n-models 1     # 原行为
    python retrain_best_model.py --mode ensemble \
        --phase1-session-id screening_20260903_191303              # select+deploy 一气呵成
    python retrain_best_model.py --mode ensemble --ensemble-select-only   # 只做 selection
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

if sys.version_info >= (3, 7):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
try:
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent

# Portable MolBlender resolution (mirrors run_screening.py).
SCRIPTS_ROOT = SCRIPT_DIR.parents[0]   # .../202609-all120-variousOP (holds tools/)
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from tools.molblender_path import ensure_molblender_on_path
ensure_molblender_on_path(SCRIPT_DIR)

# Cache: run-9.3/.mbl_cache, shared with run_screening; override via $MOLBLENDER_CACHE_DIR.
CACHE_DIR = Path(os.environ.get("MOLBLENDER_CACHE_DIR", SCRIPT_DIR.parents[0] / ".mbl_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MOLBLENDER_CACHE_DIR"] = str(CACHE_DIR)

# 与 run_screening.py 一致：MLM 表征离线可跑，允许 HF 缓存自带的 remote code
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MOLBLENDER_ALLOW_REMOTE_CODE", "1")

from tools.screening_utils import (
    setup_logging,
    get_top_models_from_phase1,
    derive_anion_group,
    build_dataset,
    clean_model_params,
)
from molblender.persistence import (
    finalize_deployment_session,
    ScreeningResultsDB,
    ScreeningSession,
    save_deployment_result,
)


def get_stage2_results_from_session(db_path: Path, session_id: str) -> list[dict]:
    """从指定 session 读所有 stage2 行（HPO 最终结果，含 best_params + grid_search_results）。"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """SELECT model_name, representation_name, hpo_score, primary_metric,
                  model_params, best_params, grid_search_results, representation_config,
                  session_id, hpo_stage, hpo_method
           FROM model_results
           WHERE session_id = ? AND hpo_score IS NOT NULL
           ORDER BY hpo_score DESC""",
        (session_id,),
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "model_name": r["model_name"],
            "representation_name": r["representation_name"],
            "hpo_score": float(r["hpo_score"]),
            "primary_metric": float(r["primary_metric"] or r["hpo_score"]),
            "model_params": json.loads(r["model_params"] or "{}"),
            "best_params": json.loads(r["best_params"] or "{}"),
            "grid_search_results": r["grid_search_results"],
            "representation_config": json.loads(r["representation_config"] or "{}"),
            "session_id": r["session_id"],
            "hpo_stage": r["hpo_stage"],
            "hpo_method": r["hpo_method"],
        })
    return results

OP_UNIT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PHASE1_DB = SCRIPT_DIR / "screening_stratified" / "screening.db"
DEFAULT_INPUT_CSV = OP_UNIT_ROOT / "data" / "reference_120.csv"
DEFAULT_CANDIDATE_CSV = OP_UNIT_ROOT / "data" / "prospective_32.csv"
DEFAULT_OUTPUT_SUBDIR = "deployment"
EXCLUDE_REPS = ["inchikey", "xmol"]

# 与 run_screening.py Stage 1 完全一致的 fold 配置（OOF 才能对齐）
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# 树/神经网络类模型：HPO 落库的 best_params 不含 random_state，裸跑每次结果不同，
# 会让 OOF/selection 不稳定（本次实测 r2/r3 两个极近 RF 成员两次跑翻转）。
# 统一注入固定 seed 保证 ensemble selection 可复现。
_SEEDED_MODELS = {
    "random_forest", "gradient_boosting", "lightgbm", "xgboost", "mlp",
    "extra_trees", "bagging", "randomforest", "xgboostregressor",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain best model(s) on all 120 OP molecules, predict 32 new N-oxides.",
    )
    p.add_argument("--phase1-db", type=Path, default=DEFAULT_PHASE1_DB)
    p.add_argument("--phase1-session-id", type=str, default=None,
                   help="Phase 1 HPO session to pick candidates from (default: largest).")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
                   help="All 120 training molecules CSV.")
    p.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV,
                   help="32 new N-oxide candidates CSV (Name, SMILES).")
    p.add_argument("--output-subdir", type=str, default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--top-n-models", type=int, default=1,
                   help="[single] Number of top Phase 1 candidates to retrain (default 1).")
    p.add_argument("--mode", type=str, choices=["single", "ensemble"], default="single",
                   help="single=原单模型(写DB); ensemble=从HPO stage2结果做selection+deploy(纯CSV).")
    p.add_argument("--ensemble-select-only", action="store_true",
                   help="[ensemble] 只做 A. selection，不做 deploy 预测。")
    p.add_argument("--ensemble-exhaustive", action="store_true",
                   help="[ensemble] 穷举所有 2-5 成员组合找全局最优（快速，OOF 已缓存）。")
    p.add_argument("--candidate-pool-size", type=int, default=None,
                   help="[已废弃] ensemble 直接用 phase1_session_id 的所有 stage2 行。")
    p.add_argument("--max-ensemble-members", type=int, default=5,
                   help="[ensemble] 成员数上限（greedy forward selection 封顶）。")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="[ensemble] 加入新模型要求 OOF MAE 至少改善的比例（0.01 = 1%）。")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# ============================================================================
# 数据加载
# ============================================================================

def load_train_data(input_csv: Path) -> pd.DataFrame:
    """Load 120 training molecules (+ Anion_Group 分层列)。"""
    df = pd.read_csv(input_csv)
    required = {"Name", "SMILES", "dG_rel_kcal_per_mol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    df = df.rename(columns={"SMILES": "standardized_smiles", "dG_rel_kcal_per_mol": "pX"})
    df["Anion_Group"] = df["Name"].map(lambda x: derive_anion_group(x))
    return df.dropna(subset=["standardized_smiles", "pX"]).reset_index(drop=True)


def load_candidate_data(candidate_csv: Path) -> pd.DataFrame:
    """Load 32 new N-oxide candidates (no labels)."""
    df = pd.read_csv(candidate_csv)
    required = {"Name", "SMILES"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate CSV missing required columns: {missing}")
    df["pX"] = np.nan  # 占位: 没真值
    return df.rename(columns={"SMILES": "standardized_smiles"}).reset_index(drop=True)


def build_stratify_key(df: pd.DataFrame) -> np.ndarray:
    """复现 Stage 1 的分层 key：Anion_Group 的 factorize（依赖 reference_120.csv 固定行序）。"""
    return pd.factorize(df["Anion_Group"].astype(str).values)[0]


# ============================================================================
# 模型 / 表征
# ============================================================================

def build_model_instance(model_info: dict, logger: logging.Logger):
    """从 Phase 1 候选信息构造 sklearn-style 模型实例。"""
    model_name = model_info["model_name"]
    model_params = clean_model_params(model_info.get("model_params", {}) or {})
    from molblender.screening.standard import get_model_registry

    registry = get_model_registry()
    model_config = registry.get_model_config(model_name)
    module_path, class_name = model_config.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    if model_name.lower() in _SEEDED_MODELS and "random_state" not in model_params:
        model_params = {**model_params, "random_state": CV_RANDOM_STATE}
    logger.info(f"Model: {model_cls.__name__} with params: {model_params}")
    return model_cls(**model_params)


def build_featurizer(rep_name: str):
    """构造一个表征 featurizer 实例 (复用 Phase 1 默认 kwargs)。"""
    from molblender.representations import FeaturizerCatalog

    catalog = FeaturizerCatalog()
    featurizer_info = catalog.get_info(rep_name)
    init_kwargs = dict(featurizer_info.default_kwargs or {})
    return featurizer_info.cls(**init_kwargs)


def load_rep_matrix(
    dataset,
    smiles_list: list[str],
    rep_name: str,
    logger: logging.Logger,
) -> np.ndarray:
    """从 MolBlender 分子级缓存读 rep 的特征矩阵（与 run_screening 共享 .mbl_cache）。

    key 基于数据集指纹（行序 hash），与 generate_full_cache / run_screening 口径一致，
    所以 MLM 等重表征命中缓存后秒读，无需重算。miss 时计算并回写缓存。
    """
    cached = dataset.load_cached_representation(rep_name)
    if cached is not None:
        data, _meta = cached
        X = np.asarray(data, dtype=float)
        logger.info(f"  [cache hit] {rep_name} shape={X.shape}")
        return X
    rep = build_featurizer(rep_name)
    logger.info(f"  [featurize] {len(smiles_list)} molecules with {rep_name}...")
    X_list = rep.featurize_many(smiles_list, show_progress=False)
    X = np.vstack([np.asarray(x, dtype=float) for x in X_list])
    dataset.cache_representation(rep_name, X)
    logger.info(f"  [cache write] {rep_name} shape={X.shape}")
    return X


def get_cv_splitter():
    """与 Stage 1 完全一致的 StratifiedKFold。"""
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)


def oof_predict(
    model_info: dict,
    X: np.ndarray,
    y: np.ndarray,
    stratify: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """固定 5-fold 生成 out-of-fold 预测（120 个，与 Stage 1 同一套 fold 划分）。"""
    splitter = get_cv_splitter()
    oof = np.empty(len(y), dtype=float)
    for fold_i, (tr_idx, te_idx) in enumerate(
        splitter.split(np.zeros(len(y)), stratify), 1
    ):
        model = build_model_instance(model_info, logger)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
        fold_mae = float(np.mean(np.abs(y[te_idx] - oof[te_idx])))
        logger.info(f"    fold {fold_i}: MAE={fold_mae:.4f} n_test={len(te_idx)}")
    return oof


# ============================================================================
# 指标
# ============================================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation（对标 Stage 1 的 primary_metric）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """单个模型的 OOF 指标：MAE + pearson。"""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mae": mae, "pearson_r": pearson_r(y_true, y_pred)}


def residual_correlation(oofs: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    """两两 OOF residual correlation matrix（diversity 分析）。"""
    keys = list(oofs.keys())
    n = len(keys)
    mat = np.ones((n, n))
    resid = {k: y_true - oofs[k] for k in keys}
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = pearson_r(resid[keys[i]], resid[keys[j]])
    return pd.DataFrame(mat, index=keys, columns=keys)


# ============================================================================
# Ensemble selection
# ============================================================================

def exhaustive_search(
    candidates: list[dict],
    oofs: dict[str, np.ndarray],
    y_true: np.ndarray,
    max_members: int,
    logger: logging.Logger,
) -> tuple[list[str], list[dict], dict[int, dict]]:
    """穷举所有 2-max_members 成员组合，找 OOF MAE 全局最优（等权平均）。

    Returns:
        best_combo: 全局最优组合的 key 列表
        all_results: 所有组合按 OOF MAE 升序排序（top 50）
        best_by_k: {n_members: 该成员数下最优组合的 row dict}
    """
    from itertools import combinations
    keys = [c["key"] for c in candidates]
    all_results = []
    best_by_k: dict[int, dict] = {}
    for k in range(2, max_members + 1):
        for combo in combinations(keys, k):
            ens_pred = np.mean([oofs[kk] for kk in combo], axis=0)
            oof_mae = float(np.mean(np.abs(y_true - ens_pred)))
            oof_r = pearson_r(y_true, ens_pred)
            row = {
                "n_members": k,
                "members": " | ".join(combo),
                "oof_mae": oof_mae,
                "oof_pearson_r": oof_r,
                "keys": list(combo),
            }
            all_results.append(row)
            cur = best_by_k.get(k)
            if cur is None or oof_mae < cur["oof_mae"]:
                best_by_k[k] = row
    all_results.sort(key=lambda x: x["oof_mae"])
    best = all_results[0]
    logger.info(f"  Exhaustive search: {len(all_results)} combinations evaluated")
    for k in sorted(best_by_k):
        logger.info(f"    best {k}-member: MAE={best_by_k[k]['oof_mae']:.4f} "
                    f"r={best_by_k[k]['oof_pearson_r']:.4f}  {best_by_k[k]['members']}")
    logger.info(f"  Best combo ({best['n_members']} members): MAE={best['oof_mae']:.4f} r={best['oof_pearson_r']:.4f}")
    return best["keys"], all_results[:50], best_by_k


def forward_selection(
    candidates: list[dict],
    oofs: dict[str, np.ndarray],
    y_true: np.ndarray,
    max_members: int,
    min_improvement: float,
    logger: logging.Logger,
) -> tuple[list[str], list[dict]]:
    """Greedy forward selection with equal-weight mean.

    从 OOF MAE 最优单模型开始，每步加入使 ensemble OOF MAE 改善最大的候选；
    相对改善 < `min_improvement`（默认 1%）或达到 `max_members` 时停止。

    Returns:
        selected: 选中的成员 key（"model|rep"）列表
        steps: 每步记录的 dict 列表（供 ensemble_selection.csv）
    """
    keys = [c["key"] for c in candidates]
    first_key = min(keys, key=lambda k: float(np.mean(np.abs(y_true - oofs[k]))))
    selected = [first_key]
    steps: list[dict] = [{
        "step": 0,
        "n_members": 1,
        "members": " | ".join(selected),
        "oof_mae": float(np.mean(np.abs(y_true - oofs[first_key]))),
        "oof_pearson_r": pearson_r(y_true, oofs[first_key]),
        "mae_improvement_pct": 0.0,
        "added": first_key,
    }]
    logger.info(f"  Step 0 (best single): {first_key} "
                f"MAE={steps[-1]['oof_mae']:.4f} r={steps[-1]['oof_pearson_r']:.4f}")

    stop_reason = None
    while len(selected) < max_members:
        best_add, best_mae, best_improvement = None, None, 0.0
        for k in keys:
            if k in selected:
                continue
            ens_pred = np.mean([oofs[kk] for kk in selected + [k]], axis=0)
            trial_mae = float(np.mean(np.abs(y_true - ens_pred)))
            improvement = (steps[-1]["oof_mae"] - trial_mae) / steps[-1]["oof_mae"]
            if best_mae is None or trial_mae < best_mae:
                best_add, best_mae, best_improvement = k, trial_mae, improvement

        if best_add is None:
            stop_reason = "no remaining candidates"
            break
        if best_improvement < min_improvement:
            stop_reason = (f"best improvement {best_improvement:.4%} < "
                           f"{min_improvement:.2%}")
            break

        selected.append(best_add)
        ens_pred = np.mean([oofs[kk] for kk in selected], axis=0)
        steps.append({
            "step": len(selected) - 1,
            "n_members": len(selected),
            "members": " | ".join(selected),
            "oof_mae": best_mae,
            "oof_pearson_r": pearson_r(y_true, ens_pred),
            "mae_improvement_pct": best_improvement,
            "added": best_add,
        })
        logger.info(f"  Step {len(selected)-1}: +{best_add} "
                    f"MAE={best_mae:.4f} (improve {best_improvement:.4%})")

    if stop_reason:
        logger.info(f"  Stopped: {stop_reason}")
        steps[-1]["stop_reason"] = stop_reason
    return selected, steps


# ============================================================================
# Deployment
# ============================================================================

def fit_and_predict(
    model_info: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """全量训练 + 预测，返回候选预测数组。"""
    model = build_model_instance(model_info, logger)
    logger.info(f"  Fitting on {len(y_train)} training samples...")
    model.fit(X_train, y_train)
    logger.info(f"  Predicting {len(X_pred)} new N-oxide candidates...")
    return np.asarray(model.predict(X_pred), dtype=float)


# ============================================================================
# ensemble 主流程（Stage3 预测写回 screening.db）
# ============================================================================

def run_ensemble(top_models: list[dict], args: argparse.Namespace,
                 output_dir: Path, logger: logging.Logger) -> int:
    """A. 用 HPO best params 重跑 OOF 做 selection → B. 选中成员全量重训 + 预测 32（纯 CSV）。"""
    train_df = load_train_data(args.input_csv)
    candidate_df = load_candidate_data(args.candidate_csv)
    y_train = np.asarray(train_df["pX"].values, dtype=float)
    stratify = build_stratify_key(train_df)
    logger.info(f"Train set: {len(train_df)} | Candidate set: {len(candidate_df)}")
    logger.info(f"Anion_Group: {train_df['Anion_Group'].value_counts().to_dict()}")

    # 表征缓存（复用 run_screening 的 .mbl_cache）
    train_ds = build_dataset(train_df)
    unique_reps = list(dict.fromkeys(m["representation_name"] for m in top_models))
    X_train_cache: dict[str, np.ndarray] = {}
    logger.info("Loading training features (shared cache per rep)...")
    train_smiles = train_df["standardized_smiles"].tolist()
    for rep_name in unique_reps:
        X_train_cache[rep_name] = load_rep_matrix(
            train_ds, train_smiles, rep_name, logger,
        )

    # ---- A. ensemble selection（用 HPO best params 重跑固定 5-fold OOF）----
    logger.info("=" * 80)
    logger.info(f"A. Ensemble selection from {len(top_models)} HPO stage2 results")
    logger.info("   (recompute OOF with HPO best params, then forward selection)")
    logger.info("=" * 80)
    oofs: dict[str, np.ndarray] = {}
    for i, m in enumerate(top_models, 1):
        logger.info(f"  [{i}/{len(top_models)}] OOF for {m['key']}")
        oofs[m["key"]] = oof_predict(
            m, X_train_cache[m["representation_name"]], y_train, stratify, logger,
        )

    per_model_rows = []
    for m in top_models:
        mm = model_metrics(y_train, oofs[m["key"]])
        per_model_rows.append({
            "model": m["model_name"],
            "rep": m["representation_name"],
            "oof_mae": mm["mae"],
            "oof_pearson_r": mm["pearson_r"],
            "hpo_score": m["hpo_score"],
        })
    per_model_df = pd.DataFrame(per_model_rows).sort_values("oof_mae").reset_index(drop=True)
    per_model_df["rank"] = np.arange(1, len(per_model_df) + 1)

    corr_df = residual_correlation(oofs, y_train)

    if args.ensemble_exhaustive:
        # 穷举所有 2-max_members 组合找全局最优
        selected, all_combos, best_by_k = exhaustive_search(
            top_models, oofs, y_train,
            max_members=args.max_ensemble_members,
            logger=logger,
        )
        # 保存 top 50 组合供选择
        combos_df = pd.DataFrame(all_combos)
        combos_df["rank"] = np.arange(1, len(combos_df) + 1)
        combos_df.to_csv(output_dir / "ensemble_exhaustive_top50.csv", index=False)
        logger.info(f"  ✅ Top 50 combinations: {output_dir / 'ensemble_exhaustive_top50.csv'}")
        # 每成员数 k 的最优组合（含 K=2，供 Fig S9 绘图）
        best_per_k_df = pd.DataFrame([
            {"n_members": k,
             "members": v["members"],
             "oof_mae": v["oof_mae"],
             "oof_pearson_r": v["oof_pearson_r"]}
            for k, v in sorted(best_by_k.items())
        ])
        best_per_k_df.to_csv(output_dir / "ensemble_exhaustive_best_per_k.csv", index=False)
        logger.info(f"  ✅ Best per k: {output_dir / 'ensemble_exhaustive_best_per_k.csv'}")
        steps = [{
            "step": 0,
            "n_members": len(selected),
            "members": " | ".join(selected),
            "oof_mae": all_combos[0]["oof_mae"],
            "oof_pearson_r": all_combos[0]["oof_pearson_r"],
            "method": "exhaustive_search",
        }]
        steps_df = pd.DataFrame(steps)
        stop_reason = "exhaustive search (global optimum)"
    else:
        # greedy forward selection
        selected, steps = forward_selection(
            top_models, oofs, y_train,
            max_members=args.max_ensemble_members,
            min_improvement=args.min_improvement,
            logger=logger,
        )
        steps_df = pd.DataFrame(steps)
        stop_reason = str(steps_df.iloc[-1].get("stop_reason", ""))

    # ensemble 整体 5-fold 指标
    ens_oof = np.mean([oofs[k] for k in selected], axis=0)
    splitter = get_cv_splitter()
    fold_metrics = []
    for fold_i, (_, te_idx) in enumerate(
        splitter.split(np.zeros(len(y_train)), stratify), 1
    ):
        fold_mae = float(np.mean(np.abs(y_train[te_idx] - ens_oof[te_idx])))
        fold_r = pearson_r(y_train[te_idx], ens_oof[te_idx])
        fold_metrics.append({"fold": fold_i, "mae": fold_mae, "pearson_r": fold_r})
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / "ensemble_fold_metrics.csv", index=False)
    logger.info(f"  Ensemble 5-fold metrics (equal-weight mean of {len(selected)} members):")
    for fm in fold_metrics:
        logger.info(f"    fold {fm['fold']}: MAE={fm['mae']:.4f} r={fm['pearson_r']:.4f}")

    steps_df.to_csv(output_dir / "ensemble_selection.csv", index=False)
    per_model_df.to_csv(output_dir / "ensemble_oof_per_model.csv", index=False)
    corr_df.to_csv(output_dir / "ensemble_residual_corr.csv", index=True)
    pd.DataFrame({
        "key": selected,
        "model": [k.split("|")[0] for k in selected],
        "rep": [k.split("|")[1] for k in selected],
    }).to_csv(output_dir / "ensemble_selected_members.csv", index=False)
    logger.info("  ✅ ensemble_selection.csv / ensemble_oof_per_model.csv / "
                "ensemble_residual_corr.csv / ensemble_selected_members.csv")

    final = steps_df.iloc[-1]
    logger.info("=" * 80)
    logger.info(f"Selected ensemble: {len(selected)} members | "
                f"OOF MAE={final['oof_mae']:.4f} | OOF pearson={final['oof_pearson_r']:.4f}")
    logger.info(f"  Members: {' | '.join(selected)}")
    if stop_reason:
        logger.info(f"  Stop reason: {stop_reason}")
    logger.info("=" * 80)

    if args.ensemble_select_only:
        return 0

    per_model_rows = []
    for m in top_models:
        mm = model_metrics(y_train, oofs[m["key"]])
        per_model_rows.append({
            "model": m["model_name"],
            "rep": m["representation_name"],
            "oof_mae": mm["mae"],
            "oof_pearson_r": mm["pearson_r"],
            "hpo_score": m["hpo_score"],
        })
    per_model_df = pd.DataFrame(per_model_rows).sort_values("oof_mae").reset_index(drop=True)
    per_model_df["rank"] = np.arange(1, len(per_model_df) + 1)

    corr_df = residual_correlation(oofs, y_train)

    if args.ensemble_exhaustive:
        # 穷举所有 2-max_members 组合找全局最优
        selected, all_combos, best_by_k = exhaustive_search(
            top_models, oofs, y_train,
            max_members=args.max_ensemble_members,
            logger=logger,
        )
        # 保存 top 50 组合供选择
        combos_df = pd.DataFrame(all_combos)
        combos_df["rank"] = np.arange(1, len(combos_df) + 1)
        combos_df.to_csv(output_dir / "ensemble_exhaustive_top50.csv", index=False)
        logger.info(f"  ✅ Top 50 combinations: {output_dir / 'ensemble_exhaustive_top50.csv'}")
        # 每成员数 k 的最优组合（含 K=2，供 Fig S9 绘图）
        best_per_k_df = pd.DataFrame([
            {"n_members": k,
             "members": v["members"],
             "oof_mae": v["oof_mae"],
             "oof_pearson_r": v["oof_pearson_r"]}
            for k, v in sorted(best_by_k.items())
        ])
        best_per_k_df.to_csv(output_dir / "ensemble_exhaustive_best_per_k.csv", index=False)
        logger.info(f"  ✅ Best per k: {output_dir / 'ensemble_exhaustive_best_per_k.csv'}")
        steps = [{
            "step": 0,
            "n_members": len(selected),
            "members": " | ".join(selected),
            "oof_mae": all_combos[0]["oof_mae"],
            "oof_pearson_r": all_combos[0]["oof_pearson_r"],
            "method": "exhaustive_search",
        }]
        steps_df = pd.DataFrame(steps)
        stop_reason = "exhaustive search (global optimum)"
    else:
        # greedy forward selection
        selected, steps = forward_selection(
            top_models, oofs, y_train,
            max_members=args.max_ensemble_members,
            min_improvement=args.min_improvement,
            logger=logger,
        )
        steps_df = pd.DataFrame(steps)
        stop_reason = str(steps_df.iloc[-1].get("stop_reason", ""))

    # ensemble 整体 5-fold 指标（验证是否真比单模型好）
    ens_oof = np.mean([oofs[k] for k in selected], axis=0)
    splitter = get_cv_splitter()
    fold_metrics = []
    for fold_i, (_, te_idx) in enumerate(
        splitter.split(np.zeros(len(y_train)), stratify), 1
    ):
        fold_mae = float(np.mean(np.abs(y_train[te_idx] - ens_oof[te_idx])))
        fold_r = pearson_r(y_train[te_idx], ens_oof[te_idx])
        fold_metrics.append({"fold": fold_i, "mae": fold_mae, "pearson_r": fold_r})
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / "ensemble_fold_metrics.csv", index=False)
    logger.info(f"  Ensemble 5-fold metrics (equal-weight mean of {len(selected)} members):")
    for fm in fold_metrics:
        logger.info(f"    fold {fm['fold']}: MAE={fm['mae']:.4f} r={fm['pearson_r']:.4f}")

    steps_df.to_csv(output_dir / "ensemble_selection.csv", index=False)
    per_model_df.to_csv(output_dir / "ensemble_oof_per_model.csv", index=False)
    corr_df.to_csv(output_dir / "ensemble_residual_corr.csv", index=True)
    pd.DataFrame({
        "key": selected,
        "model": [k.split("|")[0] for k in selected],
        "rep": [k.split("|")[1] for k in selected],
    }).to_csv(output_dir / "ensemble_selected_members.csv", index=False)
    logger.info("  ✅ ensemble_selection.csv / ensemble_oof_per_model.csv / "
                "ensemble_residual_corr.csv / ensemble_selected_members.csv")

    final = steps_df.iloc[-1]
    logger.info("=" * 80)
    logger.info(f"Selected ensemble: {len(selected)} members | "
                f"OOF MAE={final['oof_mae']:.4f} | OOF pearson={final['oof_pearson_r']:.4f}")
    logger.info(f"  Members: {' | '.join(selected)}")
    if stop_reason:
        logger.info(f"  Stop reason: {stop_reason}")
    logger.info("=" * 80)

    if args.ensemble_select_only:
        return 0

    # ---- B. final deployment ----
    logger.info("=" * 80)
    logger.info(f"B. Final deployment: retrain {len(selected)} members on all 120, predict 32")
    logger.info("=" * 80)
    cand_smiles = candidate_df["standardized_smiles"].tolist()
    cand_ds = build_dataset(candidate_df)
    X_candidate_cache: dict[str, np.ndarray] = {}
    for rep_name in unique_reps:
        X_candidate_cache[rep_name] = load_rep_matrix(
            cand_ds, cand_smiles, rep_name, logger,
        )

    member_rows: list[dict] = []
    for key in selected:
        m = next(x for x in top_models if x["key"] == key)
        rep_name = m["representation_name"]
        logger.info(f"  [{key}]")
        try:
            y_pred = fit_and_predict(
                m, X_train_cache[rep_name], y_train, X_candidate_cache[rep_name], logger,
            )
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}", exc_info=True)
            continue
        member_rows.append({"key": key, "predictions": y_pred})

    if not member_rows:
        logger.error("No members trained successfully")
        return 1

    out = pd.DataFrame({
        "Name": candidate_df["Name"],
        "SMILES": candidate_df["standardized_smiles"],
    })
    for r in member_rows:
        out[r["key"]] = r["predictions"]
    preds = np.column_stack([r["predictions"] for r in member_rows])
    out["ensemble_mean"] = preds.mean(axis=1)
    out["ensemble_sd"] = preds.std(axis=1)  # inter-model disagreement（非校准不确定性）
    out["ensemble_rank"] = out["ensemble_mean"].rank(ascending=True).astype(int)

    ens_path = output_dir / "prospective_32.csv"
    out.to_csv(ens_path, index=False)
    logger.info(f"  ✅ Ensemble predictions: {ens_path}")
    logger.info(f"  Mean range [{out['ensemble_mean'].min():.3f}, "
                f"{out['ensemble_mean'].max():.3f}] | SD [{out['ensemble_sd'].min():.3f}, "
                f"{out['ensemble_sd'].max():.3f}]")
    logger.info("  Top 6 by ensemble mean (lowest predicted dG_rel_kcal_per_mol):")
    logger.info(out.sort_values("ensemble_mean").head(6).to_string(index=False))

    (output_dir / "run_config.json").write_text(json.dumps({
        "timestamp": dt.datetime.now().isoformat(),
        "phase1_db": str(args.phase1_db),
        "phase1_session_id": top_models[0]["session_id"],
        "mode": "ensemble",
        "candidate_pool_size": args.candidate_pool_size,
        "max_ensemble_members": args.max_ensemble_members,
        "min_improvement": args.min_improvement,
        "cv_folds": CV_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "n_train": len(train_df),
        "n_candidates": len(candidate_df),
        "selected_members": selected,
        "stop_reason": stop_reason,
    }, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"Run config: {output_dir / 'run_config.json'}")
    return 0


# ============================================================================
# single 主流程（原行为，含 Stage3 deployment session 写 DB）
# ============================================================================

def run_single(top_models: list[dict], args: argparse.Namespace,
               output_dir: Path, logger: logging.Logger) -> int:
    """原单模型路径：top-n-models 各自全 120 重训 + 预测 32，写 CSV + DB。"""
    train_df = load_train_data(args.input_csv)
    candidate_df = load_candidate_data(args.candidate_csv)
    os.environ["MOLBLENDER_EXCLUDE_REPRESENTATIONS"] = ",".join(EXCLUDE_REPS)

    deployment_session_id = (
        f"deployment_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    db = ScreeningResultsDB(args.phase1_db)
    deployment_session = ScreeningSession(
        session_id=deployment_session_id,
        timestamp=dt.datetime.now().isoformat(),
        task_type="regression",
        primary_metric=str(top_models[0].get("primary_metric_name") or "pearson_r"),
        dataset_name=f"deployment:{args.candidate_csv.stem}",
        dataset_csv_path=str(args.candidate_csv),
        dataset_size=len(candidate_df),
        cv_folds=0,
        test_size=0.0,
        random_state=42,
        resolved_workers=1,
        n_jobs=1,
        train_indices=json.dumps(list(range(len(train_df)))),
        test_indices=json.dumps(list(range(len(candidate_df)))),
        cache_dir=str(CACHE_DIR),
        session_kind="deployment",
        merge_provenance=json.dumps({"source_session_id": top_models[0]["session_id"]}),
        success=False,
    )
    if not db.create_session(deployment_session):
        logger.error("Failed to create deployment session: %s", deployment_session_id)
        return 1
    logger.info("Created Stage 3 deployment session: %s", deployment_session_id)

    train_ds = build_dataset(train_df)
    cand_ds = build_dataset(candidate_df)
    train_smiles = train_df["standardized_smiles"].tolist()
    cand_smiles = candidate_df["standardized_smiles"].tolist()
    y_train = np.asarray(train_df["pX"].values, dtype=float)

    all_predictions: list[dict] = []
    for rank, model_info in enumerate(top_models[: args.top_n_models], 1):
        logger.info("=" * 80)
        logger.info(f"[{rank}/{args.top_n_models}] {model_info['model_name']} + "
                    f"{model_info['representation_name']}")
        logger.info("=" * 80)
        try:
            rep_name = model_info["representation_name"]
            X_train = load_rep_matrix(train_ds, train_smiles, rep_name, logger)
            X_pred = load_rep_matrix(cand_ds, cand_smiles, rep_name, logger)
            y_pred = fit_and_predict(model_info, X_train, y_train, X_pred, logger)
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}", exc_info=True)
            continue

        pred_df = pd.DataFrame({
            "Name": candidate_df["Name"],
            "SMILES": candidate_df["standardized_smiles"],
            "predicted_deltaG": y_pred,
        })
        safe_name = (f"predictions_32_{model_info['model_name']}_"
                     f"{model_info['representation_name']}").replace("/", "_").replace("+", "_")
        pred_csv_path = output_dir / f"{safe_name}.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        logger.info(f"  ✅ Predictions written to: {pred_csv_path}")

        result = {
            "model_name": model_info["model_name"],
            "representation_name": model_info["representation_name"],
            "model_params": clean_model_params(model_info.get("model_params", {}) or {}),
            "representation_config": model_info.get("representation_config", {}),
            "predictions": y_pred.tolist(),
            "n_train": int(len(y_train)),
            "n_test": int(len(candidate_df)),
            "phase1_session_id": model_info["session_id"],
            "phase1_primary_metric": model_info["primary_metric"],
            "phase1_hpo_score": model_info["hpo_score"],
        }
        all_predictions.append({**result, "pred_csv_path": str(pred_csv_path)})

        outcome = save_deployment_result(
            db,
            session_id=deployment_session_id,
            model_name=result["model_name"],
            representation_name=result["representation_name"],
            predictions=result["predictions"],
            train_indices=list(range(len(train_df))),
            test_indices=list(range(len(candidate_df))),
            representation_config=result.get("representation_config"),
            model_params=result.get("model_params"),
            source_session_id=result.get("phase1_session_id"),
            source_primary_metric=result.get("phase1_primary_metric"),
            source_hpo_score=result.get("phase1_hpo_score"),
            deployment_metadata={
                "input_csv": str(args.input_csv),
                "candidate_csv": str(args.candidate_csv),
                "n_train": result["n_train"],
                "n_candidates": result["n_test"],
                "prediction_csv": str(pred_csv_path),
            },
            rank=rank,
            logger=logger,
        )
        logger.info("  DB Stage 3 write: %s", getattr(outcome, "status", outcome))

    logger.info("=" * 80)
    logger.info(f"Retrain + Predict Summary: {len(all_predictions)} models trained")
    logger.info("=" * 80)
    finalize_deployment_session(
        db,
        deployment_session_id,
        n_models=len(all_predictions),
        n_representations=len({r["representation_name"] for r in all_predictions}),
        logger=logger,
    )

    (output_dir / "run_config.json").write_text(json.dumps({
        "timestamp": dt.datetime.now().isoformat(),
        "phase1_db": str(args.phase1_db),
        "phase1_session_id": top_models[0]["session_id"],
        "deployment_session_id": deployment_session_id,
        "input_csv": str(args.input_csv),
        "candidate_csv": str(args.candidate_csv),
        "n_train": len(train_df),
        "n_candidates": len(candidate_df),
        "top_n_models": args.top_n_models,
        "cache_dir": str(CACHE_DIR),
        "models_trained": [
            {k: v for k, v in r.items() if k != "predictions"} for r in all_predictions
        ],
    }, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"Run config: {output_dir / 'run_config.json'}")
    logger.info("=" * 80)
    return 0


def main() -> int:
    args = parse_args()
    output_dir = SCRIPT_DIR / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, log_name="retrain.log", force=True)

    logger.info("=" * 80)
    logger.info("Retrain best model(s) on 120 OP molecules, predict 32 new N-oxides")
    logger.info(f"Mode: {args.mode} | pool: {args.candidate_pool_size} | "
                f"max members: {args.max_ensemble_members} | min improve: {args.min_improvement}")
    logger.info("=" * 80)

    if not args.phase1_db.exists():
        logger.error(f"Phase 1 DB not found: {args.phase1_db}")
        logger.error("Run run_screening.py first to populate Stage 1+2 results.")
        return 1

    if args.mode == "ensemble":
        if not args.phase1_session_id:
            logger.error("--phase1-session-id required for ensemble mode (specify HPO session)")
            return 1
        # ensemble: 读该 session 所有 stage2 行（HPO 最终结果）
        top_models = get_stage2_results_from_session(
            args.phase1_db, args.phase1_session_id,
        )
    else:
        # single: 读 top N（保持原逻辑）
        top_models = get_top_models_from_phase1(
            args.phase1_db, args.top_n_models, session_id=args.phase1_session_id,
        )

    if not top_models:
        logger.error("No eligible Phase 1 candidates found")
        return 1
    selected_session_id = top_models[0]["session_id"]
    logger.info(f"Selected Phase 1 session: {selected_session_id}")
    logger.info(f"Loaded {len(top_models)} candidates:")
    for i, m in enumerate(top_models, 1):
        logger.info(f"  {i:2d}. {m['model_name']:25s} + {m['representation_name']:40s}: "
                    f"hpo_score={m['hpo_score']:.4f}")
    for m in top_models:
        m["key"] = f"{m['model_name']}|{m['representation_name']}"

    if args.mode == "single":
        return run_single(top_models, args, output_dir, logger)
    return run_ensemble(top_models, args, output_dir, logger)


if __name__ == "__main__":
    sys.exit(main())
