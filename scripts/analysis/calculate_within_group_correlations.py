"""Calculate within-group correlations for structure-activity relationship analysis.

This script computes Pearson correlations between molecular properties (MW, logP, TPSA)
and membrane binding free energy (ΔG) within each anion group.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from scipy.stats import pearsonr


def parse_anion_from_name(name):
    """Extract anion type from molecule name (e.g., hA001 → A_N)."""
    prefix = name[:2]
    mapping = {
        "hA": "A_N",  # Oxide
        "hN": "A_N",  # Nitro (also in A_N group)
        "hC": "C",  # Carboxylate
        "hH": "H",  # Sulfonate
        "hP": "P",  # Phosphate ester
        "hS": "S",  # Sulfate ester
    }
    return mapping.get(prefix, "Unknown")


def calc_properties(smiles):
    """Calculate molecular properties from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    return mw, logp, tpsa


def main():
    # Load data
    base_dir = Path(__file__).resolve().parents[2]
    data_file = base_dir / "data" / "reference_120.csv"
    df = pd.read_csv(data_file)

    print(f"Loaded {len(df)} molecules from reference_120.csv\n")

    # Parse anion and add properties
    df["anion"] = df["Name"].apply(parse_anion_from_name)
    df["ΔG"] = df["dG_rel_kcal_per_mol"]

    properties = df["SMILES"].apply(calc_properties)
    df["MW"] = [p[0] for p in properties]
    df["logP"] = [p[1] for p in properties]
    df["TPSA"] = [p[2] for p in properties]

    print("=== Within-Group Correlation Analysis ===\n")

    # Group by anion
    results = []
    for anion_type, group in df.groupby("anion"):
        n = len(group)
        print(f"\n{anion_type} group (n={n}):")

        if n >= 3:
            r_mw, p_mw = pearsonr(group["MW"], group["ΔG"])
            r_logp, p_logp = pearsonr(group["logP"], group["ΔG"])
            r_tpsa, p_tpsa = pearsonr(group["TPSA"], group["ΔG"])

            print(f"  MW vs ΔG:    r = {r_mw:+.3f}, p = {p_mw:.3e}")
            print(f"  logP vs ΔG:  r = {r_logp:+.3f}, p = {p_logp:.3e}")
            print(f"  TPSA vs ΔG:  r = {r_tpsa:+.3f}, p = {p_tpsa:.3e}")

            results.append(
                {
                    "anion": anion_type,
                    "n": n,
                    "r_MW": r_mw,
                    "r_logP": r_logp,
                    "r_TPSA": r_tpsa,
                    "p_MW": p_mw,
                    "p_logP": p_logp,
                    "p_TPSA": p_tpsa,
                }
            )

    # Print summary table
    print("\n" + "=" * 70)
    print("\n=== Summary Table ===\n")
    print(f"{'Anion':<10} {'n':<5} {'r(MW)':<10} {'r(logP)':<10} {'r(TPSA)':<10}")
    print("-" * 70)

    sort_order = {"C": 0, "H": 1, "P": 2, "S": 3, "A_N": 4}
    for r in sorted(results, key=lambda x: sort_order.get(x["anion"], 99)):
        sig_mw = "*" if r["p_MW"] < 0.05 else " "
        sig_logp = "*" if r["p_logP"] < 0.05 else " "
        sig_tpsa = "*" if r["p_TPSA"] < 0.05 else " "
        print(
            f"{r['anion']:<10} {r['n']:<5} {r['r_MW']:+.3f}{sig_mw}     "
            f"{r['r_logP']:+.3f}{sig_logp}     {r['r_TPSA']:+.3f}{sig_tpsa}"
        )

    print("\n* = p < 0.05 (statistically significant)")

    # A_N group breakdown
    print("\n" + "=" * 70)
    print("\n=== A_N Group Breakdown (hA vs hN) ===\n")

    a_n_group = df[df["anion"] == "A_N"]
    ha_group = a_n_group[a_n_group["Name"].str.startswith("hA")]
    hn_group = a_n_group[a_n_group["Name"].str.startswith("hN")]

    for name, subgroup in [
        ("hA (Oxide O⁻)", ha_group),
        ("hN (Nitro N-oxide)", hn_group),
    ]:
        print(f"{name} subgroup (n={len(subgroup)}):")
        if len(subgroup) >= 3:
            r_mw, p_mw = pearsonr(subgroup["MW"], subgroup["ΔG"])
            r_logp, p_logp = pearsonr(subgroup["logP"], subgroup["ΔG"])
            r_tpsa, p_tpsa = pearsonr(subgroup["TPSA"], subgroup["ΔG"])
            print(f"  MW vs ΔG:    r = {r_mw:+.3f}, p = {p_mw:.3e}")
            print(f"  logP vs ΔG:  r = {r_logp:+.3f}, p = {p_logp:.3e}")
            print(f"  TPSA vs ΔG:  r = {r_tpsa:+.3f}, p = {p_tpsa:.3e}")
            print(f"  Mean ΔG: {subgroup['ΔG'].mean():.3f} kcal/mol\n")


if __name__ == "__main__":
    main()
