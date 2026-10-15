#!/usr/bin/env python3
"""Recompute per-model OOF predictions (with RMSE) for the 5 selected deployment
ensemble members, reusing the exact fold/scaling/model logic of retrain_best_model.py.

Only used to fill in the "待核实" OOF RMSE cells of Supplementary Table S5;
does not change any selection or deployment output.
"""
import sys
import sqlite3
import json
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
    logger = rb.setup_logging(SCRIPT_DIR / "deployment" / "recompute_oof_rmse.log")

    train_df = rb.load_train_data(rb.DEFAULT_INPUT_CSV)
    stratify = rb.build_stratify_key(train_df)
    y_train = train_df["pX"].to_numpy(dtype=float)

    stage2 = rb.get_stage2_results_from_session(DB_PATH, SESSION_ID)
    by_key = {(r["model_name"], r["representation_name"]): r for r in stage2}

    train_ds = rb.build_dataset(train_df)
    smiles = train_df["standardized_smiles"].tolist()

    rows = []
    oof_all = {}
    for model_name, rep_name in SELECTED:
        info = dict(by_key[(model_name, rep_name)])
        info["model_params"] = info["best_params"] or info["model_params"]
        X = rb.load_rep_matrix(train_ds, smiles, rep_name, logger)
        oof = rb.oof_predict(info, X, y_train, stratify, logger)
        oof_all[f"{model_name}|{rep_name}"] = oof
        resid = y_train - oof
        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        r = rb.pearson_r(y_train, oof)
        rows.append({
            "model": model_name, "rep": rep_name,
            "oof_mae": mae, "oof_rmse": rmse, "oof_pearson_r": r,
        })
        print(f"{model_name}|{rep_name}: MAE={mae:.4f} RMSE={rmse:.4f} r={r:.4f}")

    ens_oof = np.mean(list(oof_all.values()), axis=0)
    resid = y_train - ens_oof
    ens_mae = float(np.mean(np.abs(resid)))
    ens_rmse = float(np.sqrt(np.mean(resid ** 2)))
    ens_r = rb.pearson_r(y_train, ens_oof)
    print(f"ENSEMBLE (pooled): MAE={ens_mae:.4f} RMSE={ens_rmse:.4f} r={ens_r:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = SCRIPT_DIR / "deployment" / "ensemble_oof_per_model_with_rmse.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
