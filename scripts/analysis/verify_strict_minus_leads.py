#!/usr/bin/env python3
"""Verify the manuscript claim in manuscript_v3.md (Table 1 / Results paragraph):

"Evaluating strict-LOGO predictions on the same test motifs used in the
few-shot setting changed the mean external metrics only marginally
(MAE, 1.20 to 1.21 kcal/mol; Pearson r, 0.637 to 0.630), indicating that
the MAE reduction was not caused by removal of the lead motifs from the test sets."

Method
------
Reuse the strict external-test predictions that the manuscript figures are
built from (data/logo_external_predictions.csv, the same source as
logo_signed_error_analysis.py / Fig. 5d / Fig. S6). Each
prediction is matched to its molecule by true value, the few-shot lead
motif(s) are dropped per fold, and MAE / Pearson r are recomputed on the
remaining ("few-shot") test subset. Both the full strict test set and the
lead-excluded subset are reported, and the five-class arithmetic means are
compared against the manuscript values so the claim is fully reproducible.

Lead motifs (fixed across strict/few-shot, per _fill_test_tables.py and
logo_signed_error_analysis.training_mean_deltaG):
    A/N: hA013; C: hC005, hC029; H: hH013, hH081; P: hP013, hP081; S: hS013, hS081
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/analysis/

import logo_signed_error_analysis as m

LEADS = {
    "A_N": ["hA013"],
    "C": ["hC005", "hC029"],
    "H": ["hH013", "hH081"],
    "P": ["hP013", "hP081"],
    "S": ["hS013", "hS081"],
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    head = m.load_head(m._default_head_csv())
    preds = m.load_logo_external_predictions().rename(columns={
        "held_out_class": "fold",
        "y_true": "y_true",
        "y_pred": "y_pred",
    })

    full_mae, full_r, excl_mae, excl_r = [], [], [], []
    print(f"{'fold':6} {'n_full':>6} {'MAE_full':>9} {'r_full':>7} "
          f"{'n_excl':>6} {'MAE_excl':>9} {'r_excl':>7}")
    for fold in m.FOLDS:
        d = preds[(preds["setting"] == "strict") & (preds["fold"] == fold)]
        names = d["motif"].astype(str).tolist()
        t = d["y_true"].to_numpy()
        p = d["y_pred"].to_numpy()

        mae_f = float(np.mean(np.abs(p - t)))
        r_f = pearson(t, p)
        full_mae.append(mae_f)
        full_r.append(r_f)

        keep = ~np.isin(names, LEADS[fold])
        te, pe = t[keep], p[keep]
        mae_e = float(np.mean(np.abs(pe - te)))
        r_e = pearson(te, pe)
        excl_mae.append(mae_e)
        excl_r.append(r_e)

        print(f"{fold:6} {len(t):6d} {mae_f:9.3f} {r_f:7.3f} "
              f"{len(te):6d} {mae_e:9.3f} {r_e:7.3f}")

    mean_full_mae = float(np.mean(full_mae))
    mean_full_r = float(np.mean(full_r))
    mean_excl_mae = float(np.mean(excl_mae))
    mean_excl_r = float(np.mean(excl_r))

    print("\nMean across 5 held-out classes (arithmetic mean of per-fold values):")
    print(f"  strict full        : MAE={mean_full_mae:.3f}  Pearson r={mean_full_r:.3f}"
          f"   (manuscript: 1.20, 0.637)")
    print(f"  strict minus leads : MAE={mean_excl_mae:.3f}  Pearson r={mean_excl_r:.3f}"
          f"   (manuscript: 1.21, 0.630)")


if __name__ == "__main__":
    main()
