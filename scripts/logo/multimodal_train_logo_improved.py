#!/usr/bin/env python3
"""Improved Leave-One-Group-Out Cross-Validation for OP_unit Dataset.

Implements improved LOGO validation where 1-2 lead molecules from the test
category are kept in training set, demonstrating practical model performance
when few examples are available.

Usage:
  python multimodal_train_logo_improved.py

After completion, view results with:
  molblender view results_logo_improved/
"""

# Suppress warnings and enable real-time logging
import os

os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['PYTHONUNBUFFERED'] = '1'

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Resolve MolBlender without a machine-specific absolute path.
current_dir = Path(__file__).parent
molblender_root = os.environ.get("MOLBLENDER_ROOT")
molblender_candidates = []
if molblender_root:
    molblender_candidates.append(Path(molblender_root).expanduser() / "src")
molblender_candidates.extend(
    [
        current_dir.parent.parent / "MolBlender" / "src",
        current_dir.parent.parent.parent / "MolBlender" / "src",
    ]
)
for src_dir in molblender_candidates:
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))
        break

# Configure warnings
from molblender.config.warnings import configure_show_once_warnings

configure_show_once_warnings()

# MolBlender imports
import argparse

from molblender.data import InputType, MolecularDataset
from molblender.models import universal_screen
from molblender.models.api.utils.reporting import (
    create_dataset_info,
    generate_all_reports,
    print_console_summary,
    print_screening_header,
)


def log_and_print(message: str):
    """Simple print-based logging with file backup."""
    print(message, flush=True)
    try:
        with open('multimodal_train_logo_improved.log', 'a', encoding='utf-8') as f:
            clean_message = message.replace('✓', '').replace('🚀', '').replace('📁', '') \
                                  .replace('🔄', '').replace('❌', '').replace('📊', '').strip()
            f.write(f"{clean_message}\n")
            f.flush()
    except:
        pass


def define_anion_groups(dataset: MolecularDataset) -> tuple[dict[str, list[int]], list[str]]:
    """Define anion category groups based on molecule naming pattern.

    Merges A and N classes as requested by user.

    Returns:
        Dict mapping group names to lists of dataset indices.

    """
    groups = {
        'A_N': [],  # Merged A (direct -O⁻) and N (N-oxide -O⁻)
        'C': [],    # Carboxylate -CO₂⁻
        'H': [],    # Sulfonate -SO₃⁻
        'P': [],    # Phosphate ester -OPO₂⁻OCH₃
        'S': []     # Sulfate ester -OSO₃⁻
    }

    # Extract molecule names from dataset
    if hasattr(dataset, 'labels') and 'Name' in dataset.labels.columns:
        names_series = dataset.labels['Name']
        # Handle both Series and DataFrame
        if hasattr(names_series, 'values'):
            names = names_series.values.flatten().tolist()
        else:
            names = list(names_series)
    else:
        # Fallback: assume sequential naming from CSV
        import pandas as pd
        df = pd.read_csv(current_dir.parent / "data" / "reference_120.csv")
        names = df['Name'].tolist()

    # Group molecules by prefix
    for idx, name in enumerate(names):
        if name.startswith('hA') or name.startswith('hN'):
            groups['A_N'].append(idx)
        elif name.startswith('hC'):
            groups['C'].append(idx)
        elif name.startswith('hH'):
            groups['H'].append(idx)
        elif name.startswith('hP'):
            groups['P'].append(idx)
        elif name.startswith('hS'):
            groups['S'].append(idx)

    # Validate grouping
    total_molecules = sum(len(indices) for indices in groups.values())
    log_and_print("\nAnion group distribution:")
    for group_name, indices in groups.items():
        log_and_print(f"  {group_name}: {len(indices)} molecules")
    log_and_print(f"  Total: {total_molecules} molecules")

    assert total_molecules == len(dataset), \
        f"Grouping error: {total_molecules} grouped vs {len(dataset)} in dataset"

    return groups, names


def select_lead_molecules(
    test_indices: list[int],
    num_leads: int,
    random_seed: int,
    strategy: str = 'auto',
    preferred_labels: Optional[list[str]] = None,
    molecule_names: Optional[list[str]] = None,
) -> tuple[list[int], list[int]]:
    """Select lead molecules from the held-out group with optional strategies."""
    if len(test_indices) <= 2:
        return [], test_indices

    rng = random.Random(random_seed)
    max_allowed = max(1, len(test_indices) - 2)  # keep at least 2 samples for testing
    desired_leads = min(max(1, num_leads), max_allowed)

    def ensure_names():
        if molecule_names is None:
            raise ValueError("molecule_names required for custom lead selection")

    def pick_by_prefix(prefix: str) -> Optional[int]:
        ensure_names()
        for idx in test_indices:
            if molecule_names[idx].startswith(prefix):
                return idx
        return None

    if preferred_labels:
        ensure_names()
        lead_indices = []
        for label in preferred_labels:
            if label not in molecule_names:
                raise ValueError(f"Lead label {label} not found in dataset")
            idx = molecule_names.index(label)
            if idx not in test_indices:
                raise ValueError(f"Lead label {label} not available in this fold")
            lead_indices.append(idx)
        remaining = [idx for idx in test_indices if idx not in lead_indices]
        return lead_indices[:desired_leads], remaining

    leads: list[int] = []

    if strategy == 'ha':
        candidate = pick_by_prefix('hA')
        if candidate is not None:
            leads = [candidate]
    elif strategy == 'hn':
        candidate = pick_by_prefix('hN')
        if candidate is not None:
            leads = [candidate]
    elif strategy == 'mixed':
        ha_idx = pick_by_prefix('hA')
        hn_idx = pick_by_prefix('hN')
        for idx in [ha_idx, hn_idx]:
            if idx is not None and idx not in leads:
                leads.append(idx)
        desired_leads = max(2, desired_leads)
        desired_leads = min(desired_leads, max_allowed)

    if len(leads) < desired_leads:
        available = [idx for idx in test_indices if idx not in leads]
        if available:
            additional = rng.sample(available, k=min(len(available), desired_leads - len(leads)))
            leads.extend(additional)

    lead_indices = leads[:desired_leads]
    remaining_test_indices = [idx for idx in test_indices if idx not in lead_indices]

    return lead_indices, remaining_test_indices


def run_improved_logo_fold(
    dataset: MolecularDataset,
    test_group_name: str,
    train_indices: list[int],
    test_indices: list[int],
    lead_indices: list[int],
    fold_output_dir: Path,
    modality_categories: list[str],
) -> dict:
    """Run one fold of improved LOGO cross-validation with lead molecules.

    Args:
        dataset: Full molecular dataset
        test_group_name: Name of the test group (e.g., 'A_N', 'C', ...)
        train_indices: Indices for training set (includes lead molecules from test group)
        test_indices: Indices for test set (test group minus leads)
        lead_indices: Indices of lead molecules from test group kept in training
        fold_output_dir: Output directory for this fold
        modality_categories: List of representation categories to screen

    Returns:
        Dictionary containing fold results and metadata

    """
    log_and_print(f"\n{'='*70}")
    log_and_print(f"FOLD: Leave-Out Group = {test_group_name} (with {len(lead_indices)} leads)")
    log_and_print(f"Train size: {len(train_indices)} (incl. {len(lead_indices)} leads) | Test size: {len(test_indices)}")
    log_and_print(f"{'='*70}\n")

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    db_file = fold_output_dir / "screening_results.db"

    # Create custom train/test split
    train_ds, test_ds = dataset.train_test_split(
        method='custom',
        train_indices=train_indices,
        test_indices=test_indices
    )

    log_and_print(f"Created custom split: {len(train_ds)} train, {len(test_ds)} test")

    # NOTE ON REPRODUCIBILITY (public release): this script was run with the
    # archived MolBlender snapshot archive/MolBlender-c0b58c6..., whose
    # multimodal pipeline does NOT propagate split_strategy='cv_only' below.
    # It therefore applied the default random 80/20 inner split (seed 42):
    # configurations are ranked by Pearson r on the internal 20% check
    # subset, and the 3-fold CV recorded below runs on the 80%
    # model-development subset as a separate diagnostic. The CURRENT
    # MolBlender checkout DOES honor split_strategy='cv_only' (pure 3-fold CV,
    # no check subset) and would NOT reproduce the paper's results; pin the
    # archived snapshot above when re-running.
    import time
    start_time = time.time()

    results = universal_screen(
        dataset=train_ds,
        target_column='dG_rel_kcal_per_mol',
        modality_categories=modality_categories,
        combinations="auto",
        task_type='regression',
        primary_metric='pearson_r',
        cv_folds=3,
        # Cross-validation only on training set (external test_ds for final evaluation)
        split_strategy='cv_only',
        # HPO configuration
        enable_hpo=True,
        hpo_stage='coarse',
        hpo_method='grid',
        top_n_for_hpo=3,
        hpo_cv_folds=3,
        # Resources
        max_cpu_cores=32,
        max_workers_per_model=4,
        random_state=42,
        verbose=2,
        # Database caching
        enable_db_storage=True,
        db_path=str(db_file),
    )

    execution_time = time.time() - start_time

    # This stage stores screening results only. The selected configuration is
    # retrained and evaluated on test_ds by retrain_logo_best_model.py.

    fold_metadata = {
        'fold_name': test_group_name,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'num_leads': len(lead_indices),
        'train_indices': train_indices,  # 完整训练集索引（包括lead分子）
        'test_indices': test_indices,    # 测试集索引（不包括lead分子）
        'lead_indices': lead_indices,    # lead分子索引
        'execution_time': execution_time,
        'success': results.get('success', False),
    }

    # Save fold metadata
    with open(fold_output_dir / 'fold_metadata.json', 'w') as f:
        json.dump(fold_metadata, f, indent=2)

    # Generate reports for this fold
    if results and results.get('success', False):
        dataset_info = create_dataset_info(
            dataset=train_ds,
            name=f'OP_unit LOGO Improved (Fold: {test_group_name}, {len(lead_indices)} leads)',
            target_column='dG_rel_kcal_per_mol',
            task_type='regression'
        )

        print_console_summary(results, dataset_info, execution_time)

        generate_all_reports(
            results=results,
            output_dir=fold_output_dir,
            report_title=f"Improved LOGO - Test Group: {test_group_name} ({len(lead_indices)} leads)",
            project_name="OP_unit Improved LOGO Cross-Validation",
            db_path=db_file
        )

    return fold_metadata


def aggregate_logo_results(all_fold_metadata: list[dict], output_dir: Path):
    """Aggregate results across all LOGO folds.

    Args:
        all_fold_metadata: List of fold metadata dictionaries
        output_dir: Root output directory

    """
    log_and_print("\n" + "="*70)
    log_and_print("IMPROVED LOGO CROSS-VALIDATION SUMMARY")
    log_and_print("="*70)

    summary = {
        'total_folds': len(all_fold_metadata),
        'folds': []
    }

    for fold_meta in all_fold_metadata:
        log_and_print(f"\nFold {fold_meta['fold_name']}:")
        log_and_print(f"  Train: {fold_meta['train_size']} ({fold_meta['num_leads']} leads) | Test: {fold_meta['test_size']}")
        log_and_print(f"  Time: {fold_meta['execution_time']:.1f}s | Success: {fold_meta['success']}")
        summary['folds'].append(fold_meta)

    # Save summary
    summary_file = output_dir / 'logo_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    log_and_print(f"\n✓ Summary saved to {summary_file}")
    log_and_print("\n" + "="*70)


def main(
    target_fold: Optional[str] = None,
    lead_strategy: str = 'auto',
    lead_labels: Optional[list[str]] = None,
    lead_random_seed: int = 42,
):
    """Main improved LOGO cross-validation workflow.

    Args:
        target_fold: If specified, only run this fold (e.g., 'S', 'P', 'C', 'H', 'A_N')
        lead_strategy: Strategy keyword controlling how lead molecules are selected
        lead_labels: Optional explicit molecule names used as leads (overrides strategy)
        lead_random_seed: Random seed for stochastic lead selection

    """
    original_dir = os.getcwd()
    os.chdir(current_dir)

    try:
        output_dir = Path("results_logo_improved")
        output_dir.mkdir(exist_ok=True)

        log_and_print(f"📁 Output directory: {output_dir.absolute()}")

        # Load dataset
        log_and_print("\n1. Loading OP_unit dataset...")
        data_file = current_dir.parent / "data" / "reference_120.csv"
        dataset = MolecularDataset.from_csv(
            filepath=str(data_file),
            input_column='SMILES',
            mol_input_type=InputType.SMILES,
            label_columns=['dG_rel_kcal_per_mol', 'Name']  # Include Name for grouping
        )
        log_and_print(f"✓ Loaded {len(dataset)} molecules")

        # Define anion groups (merge A and N)
        log_and_print("\n2. Defining anion category groups...")
        anion_groups, molecule_names = define_anion_groups(dataset)

        # Modality categories to screen (comprehensive mode)
        modality_categories = [
            # VECTOR modalities
            "fingerprints/rdkit",           # 17 fingerprints
            "fingerprints/cdk",             # 15 fingerprints
            "fingerprints/datamol",         # 16 fingerprints
            "fingerprints/deepchem",        # 6 fingerprints
            "fingerprints/spatial3d",       # 3D fingerprints (e3fp, ecif, nnscore)
            "descriptors/rdkit",            # RDKit descriptors
            "language_model",               # 10 embeddings
            "spatial/UniMol/learned_cls",   # 11 UniMol embeddings
            # STRING modality
            "sequential/string",            # 9 string representations
            # MATRIX modality
            "spatial/matrix",               # 4 matrices
            # IMAGE modality
            "image/2d",                     # 4 images
        ]

        # Run LOGO folds with lead molecules
        log_and_print("\n3. Running improved LOGO cross-validation...")
        all_fold_metadata = []
        if lead_labels:
            log_and_print(f"\n🎯 使用指定lead分子: {', '.join(lead_labels)}")
        log_and_print(f"\nLead策略: {lead_strategy} | 随机种子: {lead_random_seed}")
        random_seed = lead_random_seed

        # Filter folds if target_fold specified
        fold_items = list(anion_groups.items())
        if target_fold:
            if target_fold in anion_groups:
                fold_items = [(target_fold, anion_groups[target_fold])]
                log_and_print(f"🎯 Running only fold: {target_fold}")
            else:
                log_and_print(f"❌ Fold {target_fold} not found. Available folds: {list(anion_groups.keys())}")
                return

        for test_group_name, test_indices in fold_items:
            # Select 1-2 lead molecules from test group
            # Use 1 lead for small groups (A_N), 2 for larger groups
            num_leads = 1 if test_group_name == 'A_N' else 2
            lead_indices, remaining_test_indices = select_lead_molecules(
                test_indices=test_indices,
                num_leads=num_leads,
                random_seed=random_seed,
                strategy=lead_strategy,
                preferred_labels=lead_labels,
                molecule_names=molecule_names,
            )

            # Train indices = all other groups + lead molecules
            train_indices = [
                idx for group_name, indices in anion_groups.items()
                if group_name != test_group_name
                for idx in indices
            ] + lead_indices

            fold_output_dir = output_dir / f"fold_{test_group_name}"

            fold_metadata = run_improved_logo_fold(
                dataset=dataset,
                test_group_name=test_group_name,
                train_indices=train_indices,
                test_indices=remaining_test_indices,
                lead_indices=lead_indices,
                fold_output_dir=fold_output_dir,
                modality_categories=modality_categories,
            )

            all_fold_metadata.append(fold_metadata)

        # Aggregate results
        log_and_print("\n4. Aggregating LOGO results...")
        aggregate_logo_results(all_fold_metadata, output_dir)

        log_and_print("\n✅ Improved LOGO cross-validation completed!")
        log_and_print(f"📊 View results: molblender view {output_dir}")

    except Exception as e:
        log_and_print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved LOGO Cross-Validation for OP_unit Dataset")
    parser.add_argument("--fold", type=str, help="Target fold to run (e.g., S, P, C, H, A_N). If not specified, run all folds.")
    parser.add_argument(
        "--lead-strategy",
        type=str,
        default="auto",
        choices=["auto", "ha", "hn", "mixed"],
        help="Lead selection策略：auto随机、ha仅选hA、hn仅选hN、mixed同时选hA+hN",
    )
    parser.add_argument(
        "--lead-labels",
        type=str,
        help="使用逗号分隔的分子编号作为lead（如 hA013,hN013）。若指定则覆盖策略",
    )
    parser.add_argument(
        "--lead-random-seed",
        type=int,
        default=42,
        help="Lead随机策略使用的随机种子",
    )
    args = parser.parse_args()

    preferred_labels = None
    if args.lead_labels:
        preferred_labels = [label.strip() for label in args.lead_labels.split(',') if label.strip()]

    print_screening_header()
    log_and_print("\n🚀 Starting Improved LOGO Cross-Validation")
    log_and_print("Strategy: 1-2 lead molecules from test category kept in training\n")
    main(
        target_fold=args.fold,
        lead_strategy=args.lead_strategy,
        lead_labels=preferred_labels,
        lead_random_seed=args.lead_random_seed,
    )
