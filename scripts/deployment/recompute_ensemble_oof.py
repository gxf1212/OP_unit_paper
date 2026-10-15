#!/usr/bin/env python3
"""Regenerate ensemble_oof_predictions.csv using the exact same seeded OOF
procedure as the original ensemble selection (matches ensemble_fold_metrics.csv,
MAE=0.2817, r=0.924). The existing file on disk was produced by a separate,
unseeded re-run and disagreed with the canonical fold metrics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))              # run-9.3 (retrain_best_model)
sys.path.insert(0, str(SCRIPT_DIR.parents[0]))   # .../202609-all120-variousOP (tools)
from tools.molblender_path import ensure_molblender_on_path
ensure_molblender_on_path(SCRIPT_DIR)

import retrain_best_model as rb

DB_PATH = SCRIPT_DIR / "screening_stratified" / "screening.db"
SESSION_ID = "screening_20260903_191303"
SELECTED = [
    ("mlp", "deepchem_mol2vec"),
    ("svm_rbf", "morgan_feature_hashed_count_fp_r2_8192"),
    ("svm_rbf", "cdk_lingo"),
    ("xgboost", "datamol_secfp"),
    ("random_forest", "morgan_feature_hashed_count_fp_r2_8192"),
]


def main():
    logger = rb.setup_logging(SCRIPT_DIR / "deployment" / "recompute_ensemble_oof.log")

    train_df = rb.load_train_data(rb.DEFAULT_INPUT_CSV)
    stratify = rb.build_stratify_key(train_df)
    y_train = train_df["pX"].to_numpy(dtype=float)

    stage2 = rb.get_stage2_results_from_session(DB_PATH, SESSION_ID)
    by_key = {(r["model_name"], r["representation_name"]): r for r in stage2}

    train_ds = rb.build_dataset(train_df)
    smiles = train_df["standardized_smiles"].tolist()

    oofs = []
    for model_name, rep_name in SELECTED:
        info = dict(by_key[(model_name, rep_name)])
        info["model_params"] = info["best_params"] or info["model_params"]
        X = rb.load_rep_matrix(train_ds, smiles, rep_name, logger)
        oof = rb.oof_predict(info, X, y_train, stratify, logger)
        oofs.append(oof)

    ens_oof = np.mean(oofs, axis=0)
    resid = y_train - ens_oof
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r = rb.pearson_r(y_train, ens_oof)
    print(f"Pooled ensemble OOF: n={len(y_train)} MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f}")

    splitter = rb.get_cv_splitter()
    fold_maes, fold_rs = [], []
    for _, te_idx in splitter.split(np.zeros(len(y_train)), stratify):
        fold_maes.append(float(np.mean(np.abs(resid[te_idx]))))
        fold_rs.append(rb.pearson_r(y_train[te_idx], ens_oof[te_idx]))
    print(f"Fold-mean: MAE={np.mean(fold_maes):.4f} r={np.mean(fold_rs):.4f}")

    out_df = pd.DataFrame({
        "Name": train_df["Name"].values,
        "actual": y_train,
        "predicted": ens_oof,
    })
    out_path = SCRIPT_DIR / "deployment" / "ensemble_oof_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Overwrote: {out_path}")

    # Persist the per-member OOF predictions too, so the member-level Spearman
    # rho reported in Table 3 can be reproduced independently of this run
    # (only the ensemble predictions were written before). Columns are keyed by
    # "<model>|<rep>" in the same order as ensemble_selected_members.csv.
    member_df = pd.DataFrame({"Name": train_df["Name"].values, "actual": y_train})
    for (model_name, rep_name), oof in zip(SELECTED, oofs):
        member_df[f"{model_name}|{rep_name}"] = oof
    member_path = SCRIPT_DIR / "deployment" / "ensemble_oof_per_model_predictions.csv"
    member_df.to_csv(member_path, index=False)
    print(f"Wrote per-member OOF predictions: {member_path}")


if __name__ == "__main__":
    main()
