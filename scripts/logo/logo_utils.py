#!/usr/bin/env python3
"""Shared helpers for analyzing LOGO screening results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SUPPORTED_MODEL_KEYWORDS = (
    'xgb', 'forest', 'gradient', 'ridge', 'svm', 'lasso',
    'elastic', 'knn', 'linear', 'lightgbm', 'lgbm',
    'bagging', 'mlp', 'bayesian'
)

# Canonical LOGO sweep coverage shared by analysis and figure scripts.
LOGO_SETTINGS = ('strict', 'improved')
LOGO_FOLDS = ('A_N', 'C', 'H', 'P', 'S')


def _log(msg: str, logger: Callable[[str], None] | None):
    if logger is None:
        print(msg)
    else:
        logger(msg)


def is_supported_model(model_name: str) -> bool:
    """Return True if the model can be recreated via sklearn/xgboost/lightgbm."""
    lower = model_name.lower()
    return any(keyword in lower for keyword in SUPPORTED_MODEL_KEYWORDS)


def extract_best_supported_config(
    fold_dir: Path,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Extract the best supported (retrainable) configuration from a fold directory."""
    db_file = fold_dir / "screening_results.db"
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_file}")

    _log(f"\n读取数据库: {db_file}", logger)

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(model_results);")
    columns = {row[1] for row in cursor.fetchall()}

    metric_col = 'primary_metric' if 'primary_metric' in columns else 'score'
    hyper_cols = [c for c in ('hyperparameters', 'best_params', 'model_params') if c in columns]
    cv_col = 'cv_scores' if 'cv_scores' in columns else None

    hyper_expr = "NULL AS hyper_json"
    if hyper_cols:
        hyper_expr = f"COALESCE({', '.join(hyper_cols)}) AS hyper_json"

    cv_expr = "NULL AS cv_scores_json"
    if cv_col:
        cv_expr = f"{cv_col} AS cv_scores_json"

    query = f"""
    SELECT representation_name,
           model_name,
           {hyper_expr},
           {cv_expr},
           {metric_col} AS primary_metric
    FROM model_results
    ORDER BY {metric_col} DESC
    LIMIT 100
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError("No results found in database")

    selected = None
    skipped_models: list[str] = []

    for row in rows:
        model_name = row[1]
        if is_supported_model(model_name):
            selected = row
            break
        skipped_models.append(model_name)

    if selected is None:
        raise ValueError(
            "No supported model configurations found (top models were: "
            + ", ".join(skipped_models[:5])
            + ")"
        )

    representation, model, hyperparams_json, cv_scores_json, primary_metric = selected

    if hyperparams_json:
        try:
            hyperparams = json.loads(hyperparams_json)
        except json.JSONDecodeError:
            hyperparams = {}
    else:
        hyperparams = {}

    cv_mean = None
    cv_std = None
    if cv_scores_json:
        try:
            cv_scores = json.loads(cv_scores_json)
            if isinstance(cv_scores, list) and cv_scores:
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
        except json.JSONDecodeError:
            pass

    _log("✓ 最佳配置提取成功:", logger)
    _log(f"  - 表征: {representation}", logger)
    _log(f"  - 模型: {model}", logger)
    if cv_mean is not None:
        _log(f"  - CV均值: {cv_mean:.4f} ± {cv_std or 0:.4f}", logger)
    else:
        _log("  - CV均值: N/A", logger)

    return {
        'representation': representation,
        'model': model,
        'hyperparameters': hyperparams,
        'cv_mean': cv_mean if cv_mean is not None else 0.0,
        'cv_std': cv_std if cv_std is not None else 0.0,
        'primary_metric': primary_metric
    }


def load_fold_split(
    fold_dir: Path,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Load train/test/lead indices from metadata."""
    metadata_file = fold_dir / "fold_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")

    metadata = json.loads(metadata_file.read_text())
    train_indices = metadata.get('train_indices', [])
    test_indices = metadata.get('test_indices', [])
    lead_indices = metadata.get('lead_indices', [])

    if not train_indices or not test_indices:
        raise ValueError("train_indices or test_indices missing in metadata")

    _log("✓ 数据划分加载成功", logger)
    _log(f"  - 训练集: {len(train_indices)} 样本", logger)
    _log(f"  - 测试集: {len(test_indices)} 样本", logger)
    if lead_indices:
        _log(f"  - Lead分子: {len(lead_indices)} 个", logger)

    return train_indices, test_indices, lead_indices


def load_logo_test_predictions(results_root: Path) -> pd.DataFrame:
    """Per-sample external-test predictions of the ten retrained LOGO models.

    Reads ``results_logo_{strict,improved}/fold_*/retrained/predictions.json``
    (``test.y_true`` / ``test.y_pred``) into one long-format DataFrame with
    columns ``setting`` ('strict'/'improved'), ``fold``, ``y_true``, ``y_pred``
    and ``signed_err`` (pred - true), preserving the per-fold sample order.

    Shared by ``figures/plot_logo_performance.py`` (Fig. 5 panel d) and
    ``analysis/logo_signed_error_analysis.py``; both previously parsed the
    JSON files independently.
    """
    records: list[dict] = []
    for setting in LOGO_SETTINGS:
        for fold in LOGO_FOLDS:
            pred_path = (results_root / f"results_logo_{setting}" /
                         f"fold_{fold}" / "retrained" / "predictions.json")
            payload = json.loads(pred_path.read_text())
            y_true = np.asarray(payload["test"]["y_true"], dtype=float)
            y_pred = np.asarray(payload["test"]["y_pred"], dtype=float)
            for t, p in zip(y_true, y_pred):
                records.append({"setting": setting, "fold": fold,
                                "y_true": t, "y_pred": p,
                                "signed_err": p - t})
    return pd.DataFrame(records)
