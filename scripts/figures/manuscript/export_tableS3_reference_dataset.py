#!/usr/bin/env python3
"""Validate and print the public 120-motif reference table."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / ("head" + ".csv")
if (BASE_DIR / "data" / "reference_120.csv").exists():
    INPUT_PATH = BASE_DIR / "data" / "reference_120.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    required = {
        "Name", "SMILES", "LOGO_class", "R_group",
        "dG_rel_kcal_per_mol", "replicate_SD_kcal_per_mol",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"reference table is missing columns: {sorted(missing)}")
    if len(df) != 120 or df["Name"].duplicated().any():
        raise ValueError("reference table must contain 120 unique motifs")
    print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
