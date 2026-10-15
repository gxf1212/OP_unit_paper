#!/usr/bin/env python3
"""Export Supplementary Table S6: prospective PMF validation of the eight selected candidates.

Reconstructs the machine-readable version of the manuscript's "Supplementary
Table S6" (Results, Prospective evaluation). For the eight amine-oxide motifs
that were carried forward to umbrella-sampling PMF assessment it lists, side by
side, the frozen five-member-ensemble prediction (mean and SD across the five
members) and the PMF-derived dG_rel with its between-replicate SD across five
independent repeats, plus the library rank, the within-panel PMF rank and the
signed/absolute error. This is the same pairing plotted in Fig. 6.

Inputs:
    data/prospective_pmf_8.csv

Output: outputs/TableS6_prospective_pmf_8.csv by default

Usage:
    python export_tableS6_prospective_pmf.py [--output PATH]
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = BASE_DIR / "data" / "prospective_pmf_8.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = BASE_DIR / "manuscript" / "Supplementary_Data" / "prospective_pmf_8.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs" / "TableS6_prospective_pmf_8.csv"

SELECTED_FOR_PMF = ["N035", "N055", "N067", "N103", "N123", "N127", "N143", "N159"]


def build_table() -> pd.DataFrame:
    """Join the frozen-ensemble predictions with the prospective PMF results."""
    source = pd.read_csv(INPUT_CSV).drop(
        columns=["PMF_rank", "signed_error", "absolute_error"], errors="ignore"
    )
    merged = source.rename(columns={
        "predicted_dG_rel": "ensemble_mean",
        "PMF_dG_rel": "pmf_dG",
        "PMF_replicate_SD": "pmf_sd",
        "library_rank": "ensemble_rank",
    }).copy()
    if set(merged["Name"]) != set(SELECTED_FOR_PMF):
        missing = set(SELECTED_FOR_PMF) - set(merged["Name"])
        extra = set(merged["Name"]) - set(SELECTED_FOR_PMF)
        raise ValueError(f"PMF/prediction merge does not match the eight selected candidates (missing={sorted(missing)}, extra={sorted(extra)})")

    merged = merged.sort_values("ensemble_mean").reset_index(drop=True)
    merged["pmf_rank"] = merged["pmf_dG"].rank(method="min").astype(int)
    merged["signed_error"] = merged["ensemble_mean"] - merged["pmf_dG"]
    merged["absolute_error"] = merged["signed_error"].abs()
    return merged


def write_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Write the machine-readable table without modifying source data."""
    out = table.rename(
        columns={
            "ensemble_mean": "predicted_dG_rel",
            "ensemble_sd": "ensemble_sd",
            "ensemble_rank": "library_rank",
            "pmf_dG": "PMF_dG_rel",
            "pmf_sd": "PMF_replicate_SD",
            "pmf_rank": "PMF_rank",
        }
    )[["Name", "SMILES", "predicted_dG_rel", "ensemble_sd", "library_rank",
       "PMF_dG_rel", "PMF_replicate_SD", "PMF_rank", "signed_error", "absolute_error"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"saved: {output_path}")


def print_markdown(table: pd.DataFrame) -> None:
    """Print the markdown table body for the manuscript SI (S6)."""
    for _, r in table.iterrows():
        smi = str(r["SMILES"]).replace("|", "\\|")
        print(
            f"| {r['Name']} | {smi} | {r['ensemble_mean']:.3f} | {r['ensemble_sd']:.3f} "
            f"| {int(r['ensemble_rank'])} | {r['pmf_dG']:.3f} | {r['pmf_sd']:.3f} "
            f"| {int(r['pmf_rank'])} | {r['signed_error']:.3f} | {r['absolute_error']:.3f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    table = build_table()
    write_csv(table, args.output)
    print_markdown(table)
