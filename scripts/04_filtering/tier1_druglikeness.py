"""
tier1_druglikeness.py — Apply Lipinski + Veber + PAINS + Brenk + QED filter.

Input:  data/curated/chembl_curated_12601.csv (12,601 compounds)
Output: data/curated/tier1_passed.csv (638 compounds)

Filters (compound must pass ALL):
  - Lipinski (≤ 1 violation):  MW ≤ 500, HBD ≤ 5, HBA ≤ 10, logP ≤ 5
  - Veber:                     TPSA ≤ 140 Å², rotatable bonds ≤ 10
  - No PAINS substructures
  - No Brenk structural alerts
  - QED (quantitative estimate of drug-likeness) ≥ 0.3
"""
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "curated" / "chembl_curated_12601.csv"
OUT_PATH = ROOT / "data" / "curated" / "tier1_passed.csv"

# Build PAINS + Brenk filter catalogs
_params_pains = FilterCatalogParams()
_params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
CATALOG_PAINS = FilterCatalog(_params_pains)

_params_brenk = FilterCatalogParams()
_params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
CATALOG_BRENK = FilterCatalog(_params_brenk)


def lipinski_pass(mol) -> bool:
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Lipinski.NumHDonors(mol) > 5: violations += 1
    if Lipinski.NumHAcceptors(mol) > 10: violations += 1
    if Crippen.MolLogP(mol) > 5: violations += 1
    return violations <= 1


def veber_pass(mol) -> bool:
    return (Descriptors.TPSA(mol) <= 140.0) and (Lipinski.NumRotatableBonds(mol) <= 10)


def check_row(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"passed": False, "reason": "invalid_smiles"}
    if not lipinski_pass(mol):
        return {"passed": False, "reason": "lipinski"}
    if not veber_pass(mol):
        return {"passed": False, "reason": "veber"}
    if CATALOG_PAINS.HasMatch(mol):
        return {"passed": False, "reason": "pains"}
    if CATALOG_BRENK.HasMatch(mol):
        return {"passed": False, "reason": "brenk"}
    if QED.qed(mol) < 0.3:
        return {"passed": False, "reason": "qed"}
    return {"passed": True, "reason": None, "qed": QED.qed(mol)}


def main() -> None:
    df = pd.read_csv(IN_PATH)
    print(f"→ Applying Tier 1 drug-likeness filter to {len(df)} compounds")
    results = df["canonical_smiles"].apply(check_row).apply(pd.Series)
    df = df.join(results)
    passed = df[df["passed"]].copy()
    print(f"  Tier 1 passed: {len(passed)} / {len(df)} ({100*len(passed)/len(df):.1f}%)")
    print("  Rejections by reason:")
    for reason, n in df[~df["passed"]]["reason"].value_counts().items():
        print(f"    {reason:15s} {n:>6}")
    passed.to_csv(OUT_PATH, index=False)
    print(f"✓ Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
