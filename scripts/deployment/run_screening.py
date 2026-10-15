#!/usr/bin/env python3
"""OP_unit MolBlender screening: 5-fold Stratified CV on 120 molecules (OP reference_120.csv).

参考 tests/projects/RdRp-ligand/Ribo-IFP/0609_molblender_pred10multi/run_screening.py 改造：
  - 输入 CSV 列：Name/SMILES/dG_rel_kcal_per_mol/replicate_SD_kcal_per_mol（不含 Site_Type）
  - Anion_Group 从 Name 第2位提取（A/C/H/N/P/S，A_N 合一个 bin）
  - 5-fold Stratified K-Fold，按 Anion_Group×deltaG_bin 复合分层，保证每 fold 阴离子比例均衡
  - 默认 primary_metric=pearson_r，cv_folds=5
  - 不排除任何 multi-site 配体（数据集本来就没有 multi_site_ligands.txt 对应物）

工作流程：
  1. 全 120 个 → Stage 1 全模态 coarse screening
  2. Stage 2 HPO（--enable-hpo）→ 选 top 模型
  3. Stage 3（独立脚本 retrain_best_model.py）→ 全 120 重训，预测 32 个新 N-oxide
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

# This prospective-screening run uses a curated, locally cached model set.
# Avoid a mirror outage turning a missing optional representation into repeated
# network retries; MolFormer still needs permission to load its cached custom code.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MOLBLENDER_ALLOW_REMOTE_CODE", "1")

if sys.version_info >= (3, 7):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

try:
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent

# Portable MolBlender resolution (no machine-specific absolute paths): locate its
# ``src`` via $MOLBLENDER_ROOT, a sibling repository, or a pip-installed package.
SCRIPTS_ROOT = SCRIPT_DIR.parents[0]   # .../202609-all120-variousOP (holds tools/)
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from tools.molblender_path import ensure_molblender_on_path
ensure_molblender_on_path(SCRIPT_DIR)

# Cache directory: run-9.3/.mbl_cache, reused across runs. Override with
# $MOLBLENDER_CACHE_DIR to point at a shared cache location.
CACHE_DIR = Path(os.environ.get("MOLBLENDER_CACHE_DIR", SCRIPT_DIR.parents[0] / ".mbl_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MOLBLENDER_CACHE_DIR"] = str(CACHE_DIR)
logging.info(f"Using cache: {CACHE_DIR}")

# Shared screening utilities
OP_UNIT_ROOT = SCRIPT_DIR.parents[2]
from tools.screening_utils import (
    setup_logging,
    load_full_dataframe,
    build_dataset,
    create_stratification_bins,
    collect_results as _collect_results,
    write_summary_csv,
    log_top_results,
)

# Default input: OP_unit/data/reference_120.csv (120 molecules)
DEFAULT_INPUT_CSV = OP_UNIT_ROOT / "data" / "reference_120.csv"
DEFAULT_OUTPUT_SUBDIR = "screening_stratified"
DEFAULT_EXCLUDE_FILE = None  # OP_unit has no multi-site exclusion list

# Stage 2 HPO defaults
DEFAULT_TOP_N_FOR_HPO = 10
DEFAULT_HPO_METHOD = "grid"

# Modality categories (matches 0609's full modality set)
DEFAULT_MODALITY_CATEGORIES = [
    "fingerprints/rdkit",
    "fingerprints/datamol",
    "fingerprints/deepchem",
    "fingerprints/cdk",
    "fingerprints/spatial3d",
    "descriptors/rdkit",
    "language_model",
    "sequential/string",
    "image/2d",
    "spatial/matrix",
    "spatial/coordinates",
    "spatial/UniMol/learned_cls/unimolv1-WithHs",
    "spatial/UniMol/learned_cls/unimolv2-84m-WithHs",
    "spatial/UniMol/learned_cls/unimolv2-310m-WithHs",
    "spatial/UniMol/learned_cls/unimolv2-1.1B-WithHs",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OP_unit MolBlender screening: 5-fold Stratified CV on 120 molecules (reference_120.csv).",
    )
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
                   help="Input CSV with Name/SMILES/dG_rel_kcal_per_mol/replicate_SD_kcal_per_mol columns.")
    p.add_argument("--output-subdir", type=str, default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--max-cpu-cores", type=int, default=56)
    p.add_argument("--max-workers-per-model", type=int, default=8)
    p.add_argument("--gpu-devices", type=str, default="0,1,2,3,4,5,6,7",
                   help="Comma-separated GPU ids.")
    p.add_argument("--enable-heavy-gpu-scheduler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--heavy-jobs-per-gpu", type=int, default=1)
    p.add_argument("--heavy-max-parallel-jobs", type=int, default=None)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--primary-metric", type=str, default="pearson_r")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--combinations", type=str, default="auto",
                   choices=["auto", "comprehensive"])
    p.add_argument("--modality", action="append", dest="modalities", default=None)
    p.add_argument("--exclude-representations", action="append",
                   dest="exclude_representations", default=None)
    p.add_argument("--model", action="append", dest="models", default=None)
    p.add_argument("--skip-existing-results", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="CV folds for Stage 1. Default 5 for 120 molecules.")
    # Stage 2 HPO arguments
    p.add_argument("--enable-hpo", action="store_true",
                   help="Enable Stage 2 HPO on top N models from Stage 1.")
    p.add_argument("--hpo-stage", type=str, default="coarse",
                   choices=["coarse", "fine", "ultrafine"],
                   help="HPO stage: coarse (fast), fine (moderate), ultrafine (thorough).")
    p.add_argument("--hpo-method", type=str, default=DEFAULT_HPO_METHOD,
                   choices=["grid", "random"],
                   help="HPO method: grid (GridSearchCV) or random (RandomizedSearchCV).")
    p.add_argument("--top-n-for-hpo", type=int, default=DEFAULT_TOP_N_FOR_HPO,
                   help="Number of top performers from Stage 1 to optimize in Stage 2.")
    p.add_argument("--hpo-cv-folds", type=int, default=None,
                   help="CV folds for HPO (None=use same as Stage 1).")
    p.add_argument("--hpo-validation-strategy", type=str, default="cross_validation",
                   choices=["cross_validation", "holdout", "cv", "single_split"],
                   help="HPO validation strategy. Recommended: cross_validation or holdout.")
    return p.parse_args()


def build_anion_stratify_key(
    df: pd.DataFrame,
) -> np.ndarray:
    """生成 Stratified K-Fold 分层 key。

    OP_unit 数据集特点：
      - dG_rel_kcal_per_mol 范围 [0.07, 4.9]，中位数 2.76，几乎全为正值
      - 按 dG_rel_kcal_per_mol 三等分只能切出 1-2 个非空 bin（97/23），做不了分层
      - 阴离子 6 类：A_A=8, A_N=8, C=32, H=24, P=24, S=24
        （merge_an=True 时 A_A+A_N=A_N 共 16 个；否则独立）

    默认按 Anion_Group 单独分层：5-fold 时 A_N 期望 16/5≈3-4 个，C/H/P/S 各 5-7 个，
    每 fold 阴离子比例均衡，且每个 bin ≥8 个样本满足 StratifiedKFold n_splits 约束。

    Args:
        df: 必须含 Anion_Group 列

    Returns:
        int 数组，可直接传给 StratifiedKFold
    """
    anion = df["Anion_Group"].astype(str).values
    factorized = pd.factorize(anion)[0]
    return factorized


def load_anion_data(
    input_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """加载数据并构建分层 key。"""
    df = load_full_dataframe(input_csv)
    logging.info(f"Loaded {len(df)} molecules")
    logging.info(f"Anion_Group distribution: {df['Anion_Group'].value_counts().to_dict()}")

    stratify_array = build_anion_stratify_key(df)
    n_unique = len(np.unique(stratify_array))
    logging.info(f"Stratification key (Anion_Group): {n_unique} unique classes")
    bin_counts: dict[str, int] = {}
    for a in df["Anion_Group"].astype(str).values:
        bin_counts[a] = bin_counts.get(a, 0) + 1
    logging.info(f"Bin distribution: {bin_counts}")

    return df, stratify_array, {}


def configure_screening(args, logger):
    from molblender.screening.orchestration import (
        CoreScreeningConfig,
        ResourceConfig,
        HPOConfig,
    )

    screening_config = CoreScreeningConfig(
        task_type="regression",
        primary_metric=args.primary_metric,
        random_state=args.random_state,
        verbose=args.verbose,
        enable_feature_selection=False,
    )

    resource_config = ResourceConfig(
        auto_optimization=True,
        max_cpu_cores=args.max_cpu_cores,
        max_workers_per_model=args.max_workers_per_model,
        execution_preference="balanced",
        gpu_devices=[int(x) for x in args.gpu_devices.split(",") if x.strip()]
        if args.gpu_devices else None,
        enable_heavy_gpu_scheduler=args.enable_heavy_gpu_scheduler,
        heavy_jobs_per_gpu=args.heavy_jobs_per_gpu,
        heavy_max_parallel_jobs=args.heavy_max_parallel_jobs,
    )

    hpo_config = HPOConfig(
        enabled=args.enable_hpo,
        stage=args.hpo_stage,
        method=args.hpo_method,
        top_n_for_hpo=args.top_n_for_hpo,
        cv_folds=args.hpo_cv_folds,
        hpo_validation_strategy=args.hpo_validation_strategy,
    )

    return screening_config, resource_config, hpo_config


def run_stage1(args, df, stratify_array, output_dir, db_path, logger):
    """Run screening: Stage 1 (coarse) + optional Stage 2 (HPO on top N)."""
    from molblender.screening.orchestration import universal_screen, SplitConfig, DatabaseConfig

    logger.info("=" * 80)
    if args.enable_hpo:
        logger.info(f"Stage 1 + Stage 2 HPO: {args.hpo_stage} {args.hpo_method} search on top {args.top_n_for_hpo}")
    else:
        logger.info("Stage 1: Coarse screening (no HPO)")
    logger.info("=" * 80)
    logger.info(f"Molecules: {len(df)}")
    logger.info(f"CV folds: {args.cv_folds} (Stratified K-Fold)")
    logger.info(f"Stratification: Anion_Group×deltaG_bin composite key")

    dataset = build_dataset(df)
    logger.info(f"Dataset SMILES count: {len(dataset)}")
    screening_config, resource_config, hpo_config = configure_screening(args, logger)

    split_config = SplitConfig(
        strategy="cv_only",
        cv_folds=args.cv_folds,
        stratify=stratify_array,
    )

    db_cfg = DatabaseConfig(
        enabled=True,
        path=str(db_path),
        skip_existing_results=args.skip_existing_results,
    )

    modality_categories = args.modalities if args.modalities else DEFAULT_MODALITY_CATEGORIES
    combinations = args.combinations if not args.models else _build_debug_combos(args.models)

    os.environ["MOLBLENDER_EXCLUDE_REPRESENTATIONS"] = ",".join(
        args.exclude_representations or [
            "inchikey",
            "xmol",
            # These checkpoints are absent or incomplete in the local HF cache.
            "chemberta-zinc250k",
            "chemberta-pubchem10m",
            "smilesbert",
            "chembert",
            "molbert",
            "selformer",
        ]
    )

    logger.info(f"  Combinations: {combinations or args.combinations}")
    logger.info(f"  Modality categories: {len(modality_categories)}")
    logger.info(f"  DB: {db_path}")
    logger.info(f"  Cache: {CACHE_DIR}")
    if args.enable_hpo:
        logger.info(f"  Stage 2 HPO: {args.hpo_stage} {args.hpo_method} on top {args.top_n_for_hpo}")
        if args.hpo_cv_folds:
            logger.info(f"  HPO CV folds: {args.hpo_cv_folds}")
    else:
        logger.info(f"  Stage 2 HPO: disabled")

    results = universal_screen(
        dataset=dataset,
        target_column="pX",
        screening_config=screening_config,
        split_config=split_config,
        resource_config=resource_config,
        hpo_config=hpo_config,
        database_config=db_cfg,
        combinations=combinations,
        modality_categories=modality_categories,
    )
    return _collect_results(results, logger)


def _build_debug_combos(model_names):
    from molblender.screening.engine.base import Combination
    return [Combination(modality="vector", representations=["datamol_avalon"], models=[m])
            for m in model_names]


def main() -> int:
    args = parse_args()
    output_dir = SCRIPT_DIR / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("OP_unit MolBlender Stratified Screening (120 molecules)")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Cache Directory: {CACHE_DIR}")
    logger.info("=" * 80)

    logger.info(f"Loading CSV: {args.input_csv}")
    df, stratify_array, _ = load_anion_data(args.input_csv)
    logger.info(f"Total molecules: {len(df)}")
    logger.info(f"Stratification: {len(np.unique(stratify_array))} bins")

    db_path = output_dir / "screening.db"

    run_cfg = {
        "script": str(Path(__file__).name),
        "timestamp": dt.datetime.now().isoformat(),
        "input_csv": str(args.input_csv),
        "n_molecules": len(df),
        "anion_distribution": df["Anion_Group"].value_counts().to_dict(),
        "cv_folds": args.cv_folds,
        "stratified": True,
        "primary_metric": args.primary_metric,
        "random_state": args.random_state,
        "max_cpu_cores": args.max_cpu_cores,
        "max_workers_per_model": args.max_workers_per_model,
        "gpu_devices": args.gpu_devices,
        "modalities": args.modalities if args.modalities else DEFAULT_MODALITY_CATEGORIES,
        "skip_existing_results": args.skip_existing_results,
        "hpo_enabled": args.enable_hpo,
        "hpo_stage": args.hpo_stage if args.enable_hpo else None,
        "hpo_method": args.hpo_method if args.enable_hpo else None,
        "top_n_for_hpo": args.top_n_for_hpo if args.enable_hpo else None,
        "hpo_cv_folds": args.hpo_cv_folds,
        "hpo_validation_strategy": args.hpo_validation_strategy if args.enable_hpo else None,
        "cache_dir": str(CACHE_DIR),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n")

    results = run_stage1(args, df, stratify_array, output_dir, db_path, logger)
    write_summary_csv(results, output_dir / "summary.csv")
    best = log_top_results(results, logger, n=20, label="Screening results")

    logger.info("=" * 80)
    logger.info(f"Screening completed! Best Pearson r: {best:.4f}")
    logger.info(f"Results: {output_dir / 'summary.csv'}")
    logger.info(f"Database: {db_path}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
