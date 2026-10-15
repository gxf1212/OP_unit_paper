#!/usr/bin/env python3
"""Export Supplementary Table S7: the 32-candidate prospective amine-oxide library.

Reconstructs the full prospective candidate library described in the
manuscript (Results, Prospective prioritisation section). All 32 newly
designed amine-oxide motifs were predicted by the frozen five-model ensemble
before prospective PMF validation; eight of them (marked in the ``selected``
column) were subsequently chosen for PMF assessment. The table documents the
"32 predicted -> 8 selected" provenance of the Fig. 6 panel.

Columns per candidate: identifier, SMILES, ensemble-predicted dG_rel (mean
over the five ensemble members), ensemble SD (inter-model disagreement),
library rank (ascending predicted dG_rel within the 32-candidate library),
and the selection flag for prospective PMF evaluation. Library ranks differ
from the within-panel ranks (1-8) of Supplementary Table S6; the S7 caption
states this explicitly. (Molecular weight, logP and TPSA are intentionally
not exported here; they are recomputed from SMILES where needed, e.g. by
scripts/data_tables/build_membrane_demo_results.py.)

Input : data/prospective_32.csv                                         (frozen public table)
Output: outputs/TableS7_prospective_library_32.csv by default
        markdown table printed to stdout for pasting into the manuscript.

Usage:
    python export_tableS7_prospective_library.py [--output PATH]
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_CSV = BASE_DIR / "data" / "prospective_32.csv"
if not INPUT_CSV.exists():
    INPUT_CSV = BASE_DIR / "manuscript" / "Supplementary_Data" / "TableS7_prospective_library_32.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs" / "TableS7_prospective_library_32.csv"

SELECTED_FOR_PMF = ["N035", "N055", "N067", "N103", "N123", "N127", "N143", "N159"]


def build_table() -> pd.DataFrame:
    """Merge candidate structures with the frozen ensemble predictions and descriptors."""
    merged = pd.read_csv(INPUT_CSV)
    if len(merged) != 32:
        raise ValueError("public prospective table must contain 32 candidates")
    # Normalise column names so the script can be re-run on its own previous
    # output (which uses library_rank / predicted_dG_rel / selected_for_pmf)
    # as well as on the frozen public source (ensemble_rank / ensemble_mean /
    # selected_for_pmf) when it is available.
    merged = merged.rename(columns={
        "selected_for_pmf": "selected",
        "library_rank": "ensemble_rank",
        "predicted_dG_rel": "ensemble_mean",
    })
    # Drop RDKit descriptor columns: they are not part of the published Table S7
    # and are not consumed by any figure or table. They are recomputed from
    # SMILES where needed (e.g. scripts/data_tables/build_membrane_demo_results.py).
    merged = merged.drop(
        columns=["molecular_weight", "logP", "TPSA", "TPSA_A2"], errors="ignore"
    )
    return merged.sort_values("ensemble_rank").reset_index(drop=True)


def write_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Write the machine-readable table without modifying source data."""
    out = table.rename(
        columns={
            "ensemble_mean": "predicted_dG_rel",
            "ensemble_sd": "ensemble_sd",
            "ensemble_rank": "library_rank",
            "selected": "selected_for_pmf",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"saved: {output_path}")


def print_markdown(table: pd.DataFrame) -> None:
    """Print the markdown table body for the manuscript SI (S7).

    Mirrors the six published columns exactly (Candidate, SMILES, Predicted
    dG_rel, Ensemble SD, Library rank, Selected for PMF) so the output can be
    pasted straight into the SI without manual column trimming.
    """
    for _, r in table.iterrows():
        smi = r["SMILES"].replace("|", "\\|")
        mark = "Yes" if r["selected"] else ""
        print(
            f"| {r['Name']} | {smi} | {r['ensemble_mean']:.3f} | {r['ensemble_sd']:.3f} "
            f"| {int(r['ensemble_rank'])} | {mark} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    table = build_table()
    write_csv(table, args.output)
    print_markdown(table)
