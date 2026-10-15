#!/usr/bin/env python3
"""Analyze signed prediction errors of the LOGO external-test predictions.

For each of the ten retrained LOGO models (5 held-out classes x strict/improved),
load the per-sample external-test predictions and compare them with the
MD-derived labels. The signed error (pred - true) reveals whether a model is
systematically biased toward over- or under-prediction for a whole chemical
class, which the aggregate MAE / Pearson r hide.

Data source:
    The 120-motif reference table (data/reference_120.csv, falling back to
    data/reference_120.csv or the exported Table S3 copy) and
    data/logo_external_predictions.csv (via public_data).

Key observation the script quantifies: for a fully held-out class the model
tends to regress its predictions toward the training-set mean. For A_N, whose
true values are the lowest of the five classes, this gives a systematic
positive error (predicted membrane affinity too weak); for C/H/P the opposite.

Usage:
    python logo_signed_error_analysis.py [--output out.csv] [--head PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

from public_data import load_logo_external_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
def _default_head_csv() -> Path:
    """Locate the 120-motif reference table.

    Prefers the release CSV, then the source-workspace reference_120.csv, then the
    exported Table S3 copy (all three carry the same per-motif values and
    the column rename in load_head handles both naming schemes).
    """
    for relpath in ("reference_120.csv", "reference_120.csv"):
        candidate = PROJECT_ROOT / "data" / relpath
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / "manuscript" / "Supplementary_Data"
            / "TableS3_reference_dataset_120.csv")

# hXxxx prefix -> LOGO class / terminal subgroup.
PREFIX_TO_GROUP = {
    "hA": ("A_N", "A"),
    "hN": ("A_N", "N"),
    "hC": ("C", "C"),
    "hH": ("H", "H"),
    "hP": ("P", "P"),
    "hS": ("S", "S"),
}
FOLDS = ["A_N", "C", "H", "P", "S"]
SETTINGS = ["strict", "improved"]


def load_head(head_csv: Path) -> pd.DataFrame:
    """Load reference_120.csv and add LOGO-class / terminal-subgroup columns."""
    df = pd.read_csv(head_csv).rename(columns={
        "dG_rel_kcal_per_mol": "dG_rel_kcal_per_mol",
        "replicate_SD_kcal_per_mol": "replicate_SD_kcal_per_mol",
    })
    df[["fold", "subgroup"]] = df["Name"].str[:2].map(
        PREFIX_TO_GROUP).apply(pd.Series)
    return df


def _match_names(y_true: np.ndarray, members: pd.DataFrame) -> list[str]:
    """Map each true value to a member Name, resolving ties uniquely.

    ``members`` holds the Name/dG_rel_kcal_per_mol rows of the held-out class. Because a
    few motifs share an identical dG_rel_kcal_per_mol (e.g. hN005 and hN021), a plain
    value->Name dict would collapse them; instead each value is assigned the
    first not-yet-used matching Name.
    """
    members = members.sort_values("dG_rel_kcal_per_mol")
    used: set[str] = set()
    names: list[str] = []
    for value in y_true:
        hits: list[str] = members[np.isclose(members["dG_rel_kcal_per_mol"], value,
                                             atol=1e-6)]["Name"].astype(str).tolist()
        pick = next((h for h in hits if h not in used), None)
        if pick is None:
            pick = hits[0] if hits else f"?{value:.3f}"
        used.add(pick)
        names.append(pick)
    return names


def build_all_errors(head_csv: Path, results_root: Path) -> pd.DataFrame:
    """Assemble a per-sample table of LOGO external-test signed errors."""
    head = load_head(head_csv)
    preds = load_logo_external_predictions().rename(columns={
        "held_out_class": "fold",
        "y_true": "true",
        "y_pred": "pred",
    })
    rows: list[dict] = []
    # sort=False keeps the loader's strict-then-improved, A_N..S order
    for (setting, fold), d in preds.groupby(["setting", "fold"], sort=False):
        members: pd.DataFrame = head.loc[head["fold"] == fold]
        names = d["motif"].astype(str).tolist()
        stderr_by_name = dict(zip(members["Name"], members["replicate_SD_kcal_per_mol"]))
        for name, t, p in zip(names, d["true"], d["pred"]):
            subgroup = PREFIX_TO_GROUP.get(name[:2], ("?", "?"))[1]
            rows.append({
                "setting": setting,
                "fold": fold,
                "name": name,
                "subgroup": subgroup,
                "true": t,
                "pred": p,
                "signed_err": p - t,
                "abs_err": abs(p - t),
                "replicate_SD_kcal_per_mol": stderr_by_name.get(name, np.nan),
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signed errors per setting x fold (+ A/N subgroup split)."""
    records: list[dict] = []
    for setting, by_setting in df.groupby("setting"):
        for fold, d in by_setting.groupby("fold"):
            err = d["signed_err"]
            records.append({
                "setting": setting,
                "fold": fold,
                "subgroup": "all",
                "n": len(d),
                "mean_err": err.mean(),
                "median_err": err.median(),
                "pct_pos": 100 * (err > 0).mean(),
                "mae": np.abs(err).mean(),
                "err_min": err.min(),
                "err_max": err.max(),
            })
            if fold == "A_N":
                for subgroup, ds in d.groupby("subgroup"):
                    e = ds["signed_err"]
                    records.append({
                        "setting": setting,
                        "fold": fold,
                        "subgroup": subgroup,
                        "n": len(ds),
                        "mean_err": e.mean(),
                        "median_err": e.median(),
                        "pct_pos": 100 * (e > 0).mean(),
                        "mae": np.abs(e).mean(),
                        "err_min": e.min(),
                        "err_max": e.max(),
                    })
    return pd.DataFrame(records)


def training_mean_deltaG(head_csv: Path) -> dict[tuple[str, str], float]:
    """Mean dG_rel_kcal_per_mol of each setting's training set (per held-out fold).

    Strict A/N training = all non-A_N motifs; improved additionally removes
    the lead(s). Used to check the 'regression-to-the-mean' interpretation.
    """
    head = load_head(head_csv)
    leads = {"A_N": ["hA013"], "C": ["hC005", "hC029"], "H": ["hH013", "hH081"],
             "P": ["hP013", "hP081"], "S": ["hS013", "hS081"]}
    means: dict[tuple[str, str], float] = {}
    for setting in SETTINGS:
        for fold in FOLDS:
            if setting == "strict":
                train = head[head["fold"] != fold]
            else:
                train = head[~head["Name"].isin(leads[fold])]
            means[(setting, fold)] = float(train["dG_rel_kcal_per_mol"].mean())
    return means


def main() -> None:
    """Print the signed-error summary (optionally save the per-sample table)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, default=None,
                        help="Override the 120-motif reference table path.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the per-sample table (CSV).")
    args = parser.parse_args()

    head_csv = args.head if args.head is not None else _default_head_csv()
    df = build_all_errors(head_csv, PROJECT_ROOT)
    summary = summarize(df)

    print("=== signed error (pred - true, kcal/mol) ===")
    for setting in SETTINGS:
        print(f"\n-- {setting} --")
        print(summary[summary["setting"] == setting].to_string(index=False,
                                                               float_format="%.3f"))

    means = training_mean_deltaG(head_csv)
    print("\n=== training-set mean dG_rel_kcal_per_mol (kcal/mol) ===")
    for (setting, fold), mean in means.items():
        print(f"  {setting:8} {fold:5} {mean:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nsaved per-sample table: {args.output}")


if __name__ == "__main__":
    main()
