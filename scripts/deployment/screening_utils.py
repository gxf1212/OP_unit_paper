#!/usr/bin/env python3
"""screening_utils.py - OP_unit screening 脚本公共工具函数。

从 Ribo-IFP tools/screening_utils.py 复制并改造：
  - 输入 CSV 列从 SMILES/ddG_kcal_mol/Site_Type 改为 Name/SMILES/dG_rel_kcal_per_mol/replicate_SD_kcal_per_mol
  - Anion_Group 从 Name 第2位提取（A/C/H/N/P/S）
  - 标签列从 ddG_kcal_mol 改为 dG_rel_kcal_per_mol
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# 1. 通用工具
# ============================================================================

def setup_logging(
    output_dir: Path,
    log_name: str = "screening.log",
    force: bool = False,
) -> logging.Logger:
    """配置统一的日志格式，同时输出到文件和 stdout。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / log_name
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=force,
    )
    return logging.getLogger(__name__)


def derive_anion_group(name: str, merge_an: bool = True) -> str:
    """从 Name 第2位提取阴离子类别标签。

    Args:
        name: 化合物名（如 hA001、hN005）
        merge_an: True 时把 A 和 N 合并为 A_N 类（让 A_N = A_A + N 共 16 个，
            5-fold 时每 fold 期望 3-4 个 A_N 样本）。
            False 时区分 A_A 和 A_N 两个独立类。

    Returns:
        "A_N" / "C" / "H" / "N" / "P" / "S" / "Unknown"
    """
    if not isinstance(name, str) or len(name) < 2:
        return "Unknown"
    letter = name[1].upper()
    if letter not in {"A", "C", "H", "N", "P", "S"}:
        return "Unknown"
    if merge_an and letter in {"A", "N"}:
        return "A_N"
    if letter == "A":
        return "A_A"
    return letter


def load_full_dataframe(
    input_csv: Path,
    rename_to_molblender: bool = True,
    merge_an: bool = True,
    require_labels: bool = True,
) -> pd.DataFrame:
    """加载 CSV，验证必要列，可选重命名为 MolBlender schema。

    Args:
        input_csv: 输入 CSV 路径
        rename_to_molblender: True 时把 SMILES→standardized_smiles, dG_rel_kcal_per_mol→pX
            并从 Name 派生 Anion_Group 列
        merge_an: True 时把 A 和 N 合并为 A_N 类（每类 16 个）。
        require_labels: True时要求必有deltaG列（训练集）；False时允许无标签（测试集）

    Returns:
        清洗后的 DataFrame
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    required = {"Name", "SMILES"}
    if require_labels:
        required.add("dG_rel_kcal_per_mol")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df["Anion_Group"] = df["Name"].map(lambda x: derive_anion_group(x, merge_an=merge_an))

    if rename_to_molblender:
        rename_map = {"SMILES": "standardized_smiles"}
        if "dG_rel_kcal_per_mol" in df.columns:
            rename_map["dG_rel_kcal_per_mol"] = "pX"
        df = df.rename(columns=rename_map)
        dropna_cols = ["standardized_smiles"]
        if "pX" in df.columns:
            dropna_cols.append("pX")
        df = df.dropna(subset=dropna_cols).reset_index(drop=True)
    else:
        dropna_cols = ["SMILES"]
        if "dG_rel_kcal_per_mol" in df.columns:
            dropna_cols.append("dG_rel_kcal_per_mol")
        df = df.dropna(subset=dropna_cols).reset_index(drop=True)
    return df


def load_multi_site_ligands(exclude_file: Path) -> set[str]:
    """从文件加载要排除的 multi-site 配体名称集合。"""
    if not exclude_file.exists():
        return set()
    with open(exclude_file, "r") as f:
        excluded = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    return excluded


def build_dataset(
    df: pd.DataFrame,
    input_column: str = "standardized_smiles",
    label_columns: tuple[str, ...] = ("pX",),
    feature_columns: list[str] | None = None,
    feature_name: str = "user_features",
):
    """构建 MolecularDataset。"""
    from molblender.data import MolecularDataset

    # 只保留 df 中实际存在的 label 列
    existing_labels = [col for col in label_columns if col in df.columns]

    kwargs: dict[str, Any] = {
        "input_column": input_column,
        "label_columns": existing_labels,  # 允许空列表（无标签数据集）
    }
    if feature_columns is not None:
        kwargs["feature_columns"] = feature_columns
        kwargs["feature_name"] = feature_name
    return MolecularDataset.from_df(df, **kwargs)


def create_stratification_bins(
    df: pd.DataFrame,
    ddg_col: str = "pX",
    n_bins: int = 3,
) -> np.ndarray:
    """基于 dG_rel_kcal_per_mol 符号和量级创建分层 bin，用于 Stratified K-Fold。

    Args:
        df: 包含 dG_rel_kcal_per_mol 列的 DataFrame
        ddg_col: 标签列名（OP_unit 流程统一用 pX）
        n_bins: 3→(neg, low_pos, high_pos)，4→(neg/lo/med/hi)

    Returns:
        bin 标签数组（int factorize 后的 0/1/2...）
    """
    ddg_values = df[ddg_col].values
    if n_bins == 3:
        bin_edges = [-np.inf, 0, 2, np.inf]
        bin_labels = ["negative", "low_positive", "high_positive"]
    elif n_bins == 4:
        bin_edges = [-np.inf, 0, 2, 4, np.inf]
        bin_labels = ["negative", "low_positive", "mid_positive", "high_positive"]
    else:
        raise ValueError(f"Unsupported n_bins: {n_bins}")
    bins = pd.cut(ddg_values, bins=bin_edges, labels=bin_labels)
    stratify_array = pd.factorize(bins)[0]
    return stratify_array


# ============================================================================
# 2. 结果处理
# ============================================================================

def _extract_attr(r: Any, name: str, default: Any = None) -> Any:
    """从 dict 或 object 中取属性，兼容两种结果格式。"""
    if isinstance(r, dict):
        return r.get(name, default)
    return getattr(r, name, default)


def collect_results(results, logger: logging.Logger):
    """从 universal_screen 返回值中提取结果列表。"""
    if isinstance(results, dict):
        for key in ["all_results", "results", "formatted_results"]:
            if key in results and results[key]:
                extracted = results[key]
                logger.info(f"Extracted {len(extracted)} results from '{key}'")
                return extracted
        logger.warning(f"Dict results keys: {list(results.keys())}, but no valid results found")
        return []
    if results:
        logger.info(f"Returning {len(results)} results directly")
    return results or []


def write_summary_csv(results, output_path: Path) -> None:
    """将结果列表写入 summary CSV，兼容 dict 和 object 结果。"""
    if not results:
        output_path.write_text("model,representation,primary_metric,pearson_r,r2_score,rmse,mae\n")
        return
    rows = []
    for r in results:
        model = _extract_attr(r, "model_name", "unknown")
        representation = _extract_attr(r, "representation_name", "unknown")
        primary_metric = _extract_attr(r, "primary_metric")
        am = _extract_attr(r, "all_metrics", {}) or {}
        rows.append({
            "model": model,
            "representation": representation,
            "primary_metric": primary_metric,
            "pearson_r": am.get("pearson_r"),
            "r2_score": am.get("r2_score"),
            "rmse": am.get("rmse"),
            "mae": am.get("mae"),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def log_top_results(
    results,
    logger: logging.Logger,
    n: int = 20,
    label: str = "",
) -> float | None:
    """按 primary_metric 排序，日志输出 top N 结果，返回最佳 primary_metric。"""
    valid = [r for r in results if _extract_attr(r, "primary_metric") is not None]
    if not valid:
        logger.info(f"[{label}] No valid results to display.")
        logger.info(f"[{label}] Total results received: {len(results)}")
        if results:
            for i, r in enumerate(results[:3]):
                logger.info(f"[{label}] Sample result {i}: type={type(r)}")
        return None
    sorted_r = sorted(valid, key=lambda x: float(_extract_attr(x, "primary_metric")), reverse=True)

    has_stages = any(_extract_attr(r, "stage") is not None for r in valid)

    logger.info(f"[{label}] Top {min(n, len(sorted_r))} results by primary_metric:")
    for i, r in enumerate(sorted_r[:n], 1):
        am = _extract_attr(r, "all_metrics", {}) or {}
        pearson = am.get("pearson_r", float("nan"))
        r2 = am.get("r2_score", float("nan"))
        rmse = am.get("rmse", float("nan"))
        mae = am.get("mae", float("nan"))

        stage_prefix = ""
        if has_stages:
            stage_val = _extract_attr(r, "stage")
            if stage_val is not None:
                stage_prefix = f"[Stage {stage_val}] "

        model_name = _extract_attr(r, "model_name", "unknown")
        representation_name = _extract_attr(r, "representation_name", "unknown")

        logger.info(
            f"  {i:2d}. {stage_prefix}{model_name:30s} + {representation_name:40s} "
            f"pearson={pearson:.4f} r2={r2:.4f} rmse={rmse:.4f} mae={mae:.4f}"
        )
    best = sorted_r[0]
    return float(_extract_attr(best, "primary_metric"))


# ============================================================================
# 3. Phase2 专用
# ============================================================================

def init_phase2_db(db_path: Path) -> None:
    """Initialize the canonical MolBlender results schema."""
    from molblender.persistence import ScreeningResultsDB
    ScreeningResultsDB(db_path)


def get_top_models_from_phase1(
    db_path: Path,
    n: int,
    *,
    session_id: str | None = None,
) -> list[dict]:
    """Return distinct Phase 1 candidates ranked by HPO then CV score."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if session_id is None:
        cursor.execute("""
            SELECT session_id
            FROM model_results
            WHERE primary_metric IS NOT NULL
            GROUP BY session_id
            ORDER BY COUNT(*) DESC, MAX(id) DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row is None:
            conn.close()
            return []
        session_id = row["session_id"]

    cursor.execute("""
        SELECT model_name, representation_name, representation_config,
               model_params, model_config, primary_metric, hpo_score,
               stage, hpo_stage, hpo_method, session_id
        FROM (
            SELECT *,
                   COALESCE(hpo_score, primary_metric) AS rank_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY model_name, representation_name
                       ORDER BY COALESCE(hpo_score, primary_metric) DESC, id DESC
                   ) AS rn
            FROM model_results
            WHERE session_id = ? AND primary_metric IS NOT NULL
        )
        WHERE rn = 1
        ORDER BY rank_score DESC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        representation_config = json.loads(row["representation_config"] or "{}")
        model_params = json.loads(row["model_params"] or "{}")
        results.append({
            "model_name": row["model_name"],
            "representation_name": row["representation_name"],
            "representation_config": representation_config,
            "model_params": model_params,
            "primary_metric": row["primary_metric"],
            "hpo_score": row["hpo_score"],
            "stage": row["stage"],
            "hpo_stage": row["hpo_stage"],
            "hpo_method": row["hpo_method"],
            "session_id": row["session_id"],
        })
        if len(results) >= n:
            break
    return results


def phase2_results_exist(
    db_path: Path,
    model_name: str,
    rep_name: str | None = None,
    phase1_session_id: str | None = None,
) -> bool:
    """Check whether this candidate was already external evaluated."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    query = "SELECT all_metrics FROM model_results WHERE model_name = ? AND stage = 2"
    params: list[str] = [model_name]
    if rep_name is not None:
        query += " AND representation_name = ?"
        params.append(rep_name)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    for (metrics_json,) in rows:
        metrics = json.loads(metrics_json or "{}")
        if metrics.get("phase1_session_id") == phase1_session_id:
            return True
    return False


def save_phase2_result(db_path: Path, result: dict) -> None:
    """Persist external/PMF result through MolBlender's canonical Stage 4 API."""
    from molblender.persistence import (
        ScreeningResultsDB, ScreeningSession, save_external_evaluation_result,
    )

    db = ScreeningResultsDB(db_path)
    session_id = f"external_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.create_session(ScreeningSession(
        session_id=session_id, timestamp=dt.datetime.now().isoformat(),
        task_type="regression", primary_metric="pearson_r",
        dataset_name="external_validation", session_kind="external_validation",
        session_metadata=json.dumps({"stage": 4, "evaluation_mode": "external_validation"}),
    ))
    metrics = {k: result.get(k) for k in ("pearson_r", "spearman_rho", "r2_score", "rmse", "mae", "n_train", "n_test")}
    metrics.update({"evaluation_mode": "external_validation", "phase1_session_id": result.get("phase1_session_id"), "phase1_hpo_score": result.get("phase1_hpo_score")})
    save_external_evaluation_result(
        db, session_id=session_id, model_name=result["model_name"],
        representation_name=result["representation_name"], primary_metric=result["pearson_r"],
        predictions=result["predictions"], metrics=metrics,
        model_params=result.get("model_params"), representation_config=result.get("representation_config"),
        source_session_id=result.get("phase1_session_id"), logger=logger if "logger" in locals() else None,
    )


def clean_model_params(model_params: dict) -> dict:
    """去除 sklearn pipeline 前缀，如 mlpregressor__alpha → alpha。"""
    cleaned = {}
    for key, value in model_params.items():
        if "__" in key:
            _, actual_param = key.split("__", 1)
            cleaned[actual_param] = value
        else:
            cleaned[key] = value
    return cleaned
