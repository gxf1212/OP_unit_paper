#!/usr/bin/env python3
"""Generate complete representations for any input CSV to the shared cache.

共享脚本 — 既可以处理 120 个 OP_unit reference_120.csv，
也可以处理 32 个 prospective_32.csv，
将来还可以处理 120+32=152 个合并集。

设计：
  - MolBlender 分子级缓存基于 SMILES hash + 表征参数组合生成唯一 key
  - 同一 SMILES+表征组合一旦算过就永不再算，与"哪个数据集、多少分子"无关
  - 因此：先分别缓存 120 个和 32 个，未来跑 152 个合并集时直接命中两者交集

用法:
    # 默认跑 120 个 OP reference_120.csv
    python tools/generate_full_cache.py

    # 处理 32 个新 N-oxide 候选
    python tools/generate_full_cache.py \\
        --input-csv OP_unit/data/prospective_32.csv

    # 处理 120+32 合并
    python tools/generate_full_cache.py \\
        --input-csv OP_unit/data/all_noxides.csv

输入 CSV 列要求:
    - 必须含 Name, SMILES 列；缓存候选集时不要求 dG_rel_kcal_per_mol
    - 额外列（replicate_SD_kcal_per_mol、Anion_Group 等）会被忽略
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

# Cache generation must not query HF for optional models that are not present
# locally. MolFormer uses cached custom code and therefore needs this opt-in.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MOLBLENDER_ALLOW_REMOTE_CODE", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate complete representations to shared cache.",
    )
    p.add_argument("--input-csv", type=Path, required=True,
                   help="CSV with Name/SMILES/dG_rel_kcal_per_mol columns.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Cache directory (default: OP_unit/.mbl_cache).")
    p.add_argument("--max-cpu-cores", type=int, default=56)
    p.add_argument("--max-workers-per-model", type=int, default=8)
    p.add_argument("--gpu-devices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--modality", action="append", dest="modalities", default=None)
    p.add_argument("--exclude-representations", action="append",
                   dest="exclude_representations", default=None)
    p.add_argument("--dry-list-representations", action="store_true",
                   help="Expand modality categories into representation names and exit.")
    p.add_argument("--featurize-n-workers", type=int, default=8,
                   help="Parallel workers for CPU featurizers (RDKit/DeepChem/Mordred).")
    return p.parse_args()


DEFAULT_MODALITY_CATEGORIES = [
    "fingerprints/rdkit",
    "fingerprints/datamol",
    "fingerprints/deepchem",
    "fingerprints/cdk",
    "fingerprints/spatial3d",
    "descriptors/rdkit",
    "descriptors/mordred",
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

# 表征 xmol（git clone 失败）、plip_interaction_fingerprint（无蛋白结构）、
# UniMol-1.1B（subprocess Queue.get 卡死）需要排除
MANUAL_SKIP_REPS = {
    "xmol",
    "UniMol-CLS-unimolv2-1.1B-WithHs",
    # Not available as complete local checkpoints in this environment.
    "chemberta-zinc250k",
    "chemberta-pubchem10m",
    "smilesbert",
    "chembert",
    "molbert",
    "selformer",
}


def setup_environment(input_csv: Path, cache_dir: Path | None) -> Path:
    """配置 sys.path / MOLBLENDER_CACHE_DIR，返回 cache 目录路径。"""
    # SCRIPTS_ROOT: 放 tools/ 的目录（.../202609-all120-variousOP）。
    here = Path(__file__).resolve().parent          # tools/
    scripts_root = here.parents[0]                  # .../202609-all120-variousOP
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from tools.molblender_path import ensure_molblender_on_path
    ensure_molblender_on_path(scripts_root)

    # Cache dir: default to <project>/.mbl_cache, shared with run_screening.py /
    # retrain_best_model.py. Override with $MOLBLENDER_CACHE_DIR or the argument.
    if cache_dir is None:
        cache_dir = Path(os.environ.get("MOLBLENDER_CACHE_DIR", scripts_root / ".mbl_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MOLBLENDER_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def load_csv(input_csv: Path):
    """加载 CSV。缓存生成只需 SMILES 列表，不需要 MolecularDataset。"""
    from tools.screening_utils import load_full_dataframe

    df = load_full_dataframe(input_csv, require_labels=False)
    return df


def main() -> int:
    args = parse_args()
    cache_dir = setup_environment(args.input_csv, args.cache_dir)
    start_time = dt.datetime.now()

    logger = logging.getLogger("generate_full_cache")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(cache_dir / "generate_full_cache.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * 80)
    logger.info("OP_unit Full Cache Generation")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Cache Directory: {cache_dir}")
    logger.info("=" * 80)

    df = load_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} molecules")
    if "Anion_Group" in df.columns:
        logger.info(f"Anion_Group distribution: {df['Anion_Group'].value_counts().to_dict()}")

    modality_categories = args.modalities if args.modalities else DEFAULT_MODALITY_CATEGORIES
    logger.info(f"Modality categories: {len(modality_categories)}")
    for mc in modality_categories:
        logger.info(f"  - {mc}")

    os.environ["MOLBLENDER_EXCLUDE_REPRESENTATIONS"] = ",".join(
        args.exclude_representations or [
            "inchikey",
            "xmol",
            "plip_interaction_fingerprint",
            "chemberta-zinc250k",
            "chemberta-pubchem10m",
            "smilesbert",
            "chembert",
            "molbert",
            "selformer",
        ],
    )
    logger.info(f"Excluded representations: {os.environ['MOLBLENDER_EXCLUDE_REPRESENTATIONS']}")

    from molblender.screening.orchestration import ResourceConfig
    from molblender.models.api.infrastructure.representation_resource_policy import (
        RepresentationResourcePolicy,
        RepresentationGpuSlot,
    )

    resource_config = ResourceConfig(
        auto_optimization=True,
        max_cpu_cores=args.max_cpu_cores,
        max_workers_per_model=args.max_workers_per_model,
        execution_preference="balanced",
        gpu_devices=[int(x) for x in args.gpu_devices.split(",") if x.strip()]
        if args.gpu_devices else None,
        enable_heavy_gpu_scheduler=True,
        heavy_jobs_per_gpu=1,
    )

    gpu_slots = [
        RepresentationGpuSlot(gpu_id=gid, device_str=f"cuda:{gid}")
        for gid in (resource_config.gpu_devices or [])
    ]
    resource_policy = RepresentationResourcePolicy(
        gpu_slots=gpu_slots,
        heavy_jobs_per_gpu=1,
        enabled=bool(gpu_slots),
        heavy_max_parallel_jobs=None,
    )

    logger.info(f"GPU devices: {args.gpu_devices}")
    logger.info(f"Max CPU cores: {args.max_cpu_cores}")
    logger.info(f"Max workers per model: {args.max_workers_per_model}")
    logger.info(f"Featurize n_workers: {args.featurize_n_workers}")
    logger.info(f"GPU resource policy: {resource_policy.max_parallel_jobs()} max parallel jobs")

    from molblender.api.representations import list_featurizers
    from molblender.screening.orchestration.config import _expand_modality_categories

    available = list_featurizers()
    logger.info(f"Available representations: {len(available)}")

    excluded_reps = MANUAL_SKIP_REPS | {
        item.strip()
        for item in os.environ["MOLBLENDER_EXCLUDE_REPRESENTATIONS"].split(",")
        if item.strip()
    }
    expanded_reps = [
        representation
        for representation in _expand_modality_categories(modality_categories)
        if representation not in excluded_reps
    ]
    logger.info(f"Representations to compute: {len(expanded_reps)}")
    for r in expanded_reps:
        logger.info(f"  - {r}")

    if args.dry_list_representations:
        logger.info(f"=== DRY LIST: {len(expanded_reps)} representations would be computed ===")
        for r in sorted(expanded_reps):
            logger.info(f"  {r}")
        return 0

    # Cache keys are scoped by the full-dataset fingerprint, not per-molecule,
    # so "already cached" must be checked against THIS dataset's fingerprint.
    from tools.screening_utils import build_dataset

    dataset = build_dataset(df)

    from molblender.data.cache.fingerprint import compute_dataset_fingerprint

    current_dataset_id = compute_dataset_fingerprint(dataset.molecules)
    logger.info(f"Dataset fingerprint: {current_dataset_id}")

    cache_metadata_path = cache_dir / "cache_metadata.json"
    cached_reps_for_dataset: set[str] = set()
    if cache_metadata_path.exists():
        with open(cache_metadata_path, "r") as f:
            metadata = json.load(f)
        for entry in metadata.get("representations", {}).values():
            if entry.get("dataset_id") == current_dataset_id:
                cached_reps_for_dataset.add(entry.get("representation_name"))

    missing_reps = [r for r in expanded_reps if r not in cached_reps_for_dataset]
    logger.info(
        f"Dataset-scoped cached: {len(cached_reps_for_dataset)}, "
        f"Missing: {len(missing_reps)}"
    )

    if missing_reps:
        compute_reps = [r for r in missing_reps if r not in MANUAL_SKIP_REPS]
        skipped_reps = [r for r in missing_reps if r in MANUAL_SKIP_REPS]
        if skipped_reps:
            logger.info(f"Manually skipping {len(skipped_reps)} problematic representations:")
            for r in skipped_reps:
                logger.info(f"  - {r}")

        logger.info(f"Computing {len(compute_reps)} missing representations...")

        # GPU precompute
        logger.info("Phase 1: GPU precompute for GPU-heavy representations...")
        from molblender.screening.engine.data.gpu_precompute import precompute_gpu_representations

        try:
            precompute_gpu_representations(
                dataset=dataset,
                representation_names=compute_reps,
                resource_policy=resource_policy,
            )
            logger.info("✓ GPU precompute completed")
        except Exception as e:
            logger.warning(f"GPU precompute failed: {e}. Falling back to sequential computation.")

        # CPU-light
        gpu_precomputed = set(dataset._features.columns) if hasattr(dataset, "_features") else set()
        cpu_reps = [r for r in compute_reps if r not in gpu_precomputed]

        if cpu_reps:
            logger.info(f"Phase 2: Computing {len(cpu_reps)} CPU-light representations...")
            for i, rep_name in enumerate(cpu_reps, 1):
                logger.info(f"[{i}/{len(cpu_reps)}] Computing {rep_name}...")
                try:
                    dataset.add_features(rep_name, feature_names=rep_name, use_cache=True)
                    logger.info(f"  ✓ {rep_name} computed")
                except Exception as e:
                    logger.error(f"  ✗ {rep_name} failed: {e}")
        else:
            logger.info("All representations computed via GPU precompute!")
    else:
        logger.info("All representations already cached!")

    elapsed = (dt.datetime.now() - start_time).total_seconds()
    logger.info(f"Cache generation completed in {elapsed:.1f} seconds")

    if cache_metadata_path.exists():
        with open(cache_metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Final cache representations: {len(metadata.get('representations', {}))}")

        from collections import Counter
        mol_counts = Counter(r.get("n_molecules", 0) for r in metadata.get("representations", {}).values())
        for n_mol, count in sorted(mol_counts.items()):
            logger.info(f"  {n_mol} molecules: {count} representations")

    logger.info("=" * 80)
    logger.info(f"Cache generation complete! Cache at: {cache_dir}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
