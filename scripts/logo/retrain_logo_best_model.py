#!/usr/bin/env python3
"""从LOGO结果中提取最佳模型配置，重新训练并验证复现性.

功能：
1. 从SQLite数据库读取最佳模型配置（表征+模型+超参数）
2. 在相同的train/test split上重新训练
3. 验证能复现LOGO筛选时的性能
4. 保存训练好的模型和完整预测结果
5. 生成用于后续分析和画图的数据文件

Usage:
  # 从Improved LOGO的某个fold提取最佳配置并重新训练
  python retrain_best_model.py --fold A_N --logo_type improved

  # 或者手动指定配置
  python retrain_best_model.py --representation coulomb_matrix --model xgboost --fold A_N
"""

import os

os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Setup paths
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from molblender.config.warnings import configure_show_once_warnings

configure_show_once_warnings()

from logo_utils import extract_best_supported_config
from logo_utils import load_fold_split as load_fold_split_from_meta

from molblender.data import InputType, MolecularDataset


def log_and_print(message: str):
    """Simple print-based logging with file backup."""
    print(message, flush=True)
    try:
        with open('retrain_best_model.log', 'a', encoding='utf-8') as f:
            clean_message = message.replace('✓', '').replace('🚀', '').replace('📁', '') \
                                  .replace('🔄', '').replace('❌', '').replace('📊', '').strip()
            f.write(f"{clean_message}\n")
            f.flush()
    except:
        pass


def extract_best_config_from_db(fold_dir: Path) -> dict[str, Any]:
    """Compatibility wrapper for shared logo utils."""
    config = extract_best_supported_config(fold_dir, logger=log_and_print)
    log_and_print(f"  - 超参数: {json.dumps(config['hyperparameters'], indent=4)}")
    return config


def load_fold_split(fold_dir: Path) -> tuple[list, list, list]:
    """Wrapper around shared metadata loader with logging."""
    return load_fold_split_from_meta(fold_dir, logger=log_and_print)


def retrain_with_config(
    dataset_path: str,
    config: dict[str, Any],
    train_indices: list,
    test_indices: list,
    output_dir: Path,
    target_column: str = 'dG_rel_kcal_per_mol'
) -> dict[str, Any]:
    """用指定配置重新训练模型（使用新的export API）.

    Args:
        dataset_path: 数据集文件路径
        config: 模型配置（表征+模型+超参数）
        train_indices: 训练集索引
        test_indices: 测试集索引
        output_dir: 输出目录
        target_column: 目标变量列名

    Returns:
        训练和评估结果

    """
    from molblender.models.api.export import (
        generate_reproduction_code,
        print_model_configuration,
        recreate_model_from_config,
    )

    log_and_print(f"\n{'='*70}")
    log_and_print("开始重新训练最佳模型")
    log_and_print(f"{'='*70}\n")

    # 打印模型配置（让用户看到详细信息）
    print_model_configuration(
        representation=config['representation'],
        model_name=config['model'],
        hyperparameters=config['hyperparameters'],
        metrics={
            'CV Mean': config.get('cv_mean', 'N/A'),
            'CV Std': config.get('cv_std', 'N/A')
        }
    )

    # 使用新的API重新训练模型
    results = recreate_model_from_config(
        representation=config['representation'],
        model_name=config['model'],
        hyperparameters=config['hyperparameters'],
        dataset_path=dataset_path,
        target_column=target_column,
        train_indices=train_indices,
        test_indices=test_indices,
        output_dir=output_dir,
        verbose=True
    )

    # 生成可复现代码并保存
    code = generate_reproduction_code(
        representation=config['representation'],
        model_name=config['model'],
        hyperparameters=config['hyperparameters'],
        dataset_path=dataset_path,
        target_column=target_column,
        train_indices=train_indices,
        test_indices=test_indices
    )

    code_file = output_dir / "reproduce_model.py"
    with open(code_file, 'w') as f:
        f.write(code)
    log_and_print(f"✓ 可复现代码已保存: {code_file}")
    log_and_print(f"  可以直接运行: python {code_file}")

    return results


def verify_reproducibility(original_metric: float, retrained_metric: float, tolerance: float = 0.01):
    """验证重训练结果能否复现原始性能.

    Args:
        original_metric: LOGO筛选时的性能
        retrained_metric: 重新训练后的性能
        tolerance: 允许的误差范围

    """
    diff = abs(original_metric - retrained_metric)
    is_reproducible = diff <= tolerance

    log_and_print(f"\n{'='*70}")
    log_and_print("可复现性验证")
    log_and_print(f"{'='*70}")
    log_and_print(f"原始性能 (LOGO筛选): {original_metric:.4f}")
    log_and_print(f"重训练性能:         {retrained_metric:.4f}")
    log_and_print(f"差异:              {diff:.4f}")
    log_and_print(f"可复现: {'✓ YES' if is_reproducible else '✗ NO'} (tolerance={tolerance})")
    log_and_print(f"{'='*70}\n")

    return is_reproducible


def main():
    parser = argparse.ArgumentParser(description='从LOGO结果重新训练最佳模型')
    parser.add_argument('--fold', required=True, choices=['A_N', 'C', 'H', 'P', 'S'],
                       help='选择哪个fold的结果')
    parser.add_argument('--logo_type', default='improved', choices=['strict', 'improved'],
                       help='选择LOGO类型')
    parser.add_argument('--representation', help='手动指定表征（可选）')
    parser.add_argument('--model', help='手动指定模型（可选）')

    args = parser.parse_args()

    original_dir = os.getcwd()
    os.chdir(current_dir)

    try:
        # 1. 定位fold结果目录
        logo_dir = current_dir / f"results_logo_{args.logo_type}"
        fold_dir = logo_dir / f"fold_{args.fold}"

        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        log_and_print(f"📁 Fold目录: {fold_dir}")

        # 2. 提取最佳配置（或使用手动指定）
        if args.representation and args.model:
            log_and_print("\n使用手动指定配置:")
            log_and_print(f"  - 表征: {args.representation}")
            log_and_print(f"  - 模型: {args.model}")
            config = {
                'representation': args.representation,
                'model': args.model,
                'hyperparameters': {},  # 需要从数据库查询
            }
        else:
            config = extract_best_config_from_db(fold_dir)

        # 3. 从fold_metadata.json加载train/test索引
        log_and_print("\n加载数据划分...")
        train_indices, test_indices, _lead_indices = load_fold_split(fold_dir)

        # 4. 重新训练
        log_and_print("\n准备重新训练...")
        data_file = current_dir.parent / "data" / "reference_120.csv"
        output_dir = fold_dir / "retrained"
        output_dir.mkdir(exist_ok=True)

        results = retrain_with_config(
            dataset_path=str(data_file),
            config=config,
            train_indices=train_indices,
            test_indices=test_indices,
            output_dir=output_dir,
            target_column='dG_rel_kcal_per_mol'
        )

        # 5. 验证可复现性
        log_and_print("\n验证可复现性...")
        original_cv = config.get('primary_metric', config.get('cv_mean', 0.0))
        retrained_test = results['test_metrics']['pearson_r']

        verify_reproducibility(original_cv, retrained_test, tolerance=0.1)

        log_and_print("\n✅ 重新训练完成!")
        log_and_print(f"📊 结果保存在: {output_dir}")
        log_and_print("\n包含文件:")
        log_and_print("  - trained_model.pkl: 训练好的模型对象")
        log_and_print("  - predictions.json: 预测结果和指标")
        log_and_print("  - reproduce_model.py: 可执行的重现代码")
        log_and_print("\n下一步: 使用predictions.json生成图表")

    except Exception as e:
        log_and_print(f"\n💥 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        os.chdir(original_dir)

    return 0


if __name__ == "__main__":
    exit(main())
