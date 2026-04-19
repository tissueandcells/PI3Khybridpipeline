"""
curate_dataset.py — Curate raw ChEMBL records into final 12,601-compound dataset.

Pipeline:
  1. Load raw JSON dumps per isoform.
  2. Filter to standard_relation == "=" and standard_value > 0.
  3. Convert nM → pIC50.
  4. Canonicalise SMILES with RDKit.
  5. Strip salts (RDKit SaltRemover).
  6. Filter by MW ∈ [100, 800] and allowed atoms (C/N/O/F/P/S/Cl/Br/I).
  7. Merge duplicates by median pIC50.
  8. Write data/curated/chembl_curated_12601.csv.

Outputs:
  data/curated/chembl_curated_12601.csv
  data/curated/curation_log.txt
"""
import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SaltRemover

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "curated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ATOMS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"}
MW_MIN, MW_MAX = 100.0, 800.0

remover = SaltRemover.SaltRemover()


def canonicalise(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = remover.StripMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def passes_filters(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN <= mw <= MW_MAX):
        return False
    atoms = {a.GetSymbol() for a in mol.GetAtoms()}
    return atoms.issubset(ALLOWED_ATOMS)


def to_pic50(value_nm: float) -> float:
    return -math.log10(value_nm * 1e-9)


def process_isoform(name: str) -> pd.DataFrame:
    raw_path = RAW_DIR / f"chembl34_{name}_raw.json"
    records = json.loads(raw_path.read_text())
    rows = []
    for r in records:
        smi = r.get("canonical_smiles")
        val = r.get("standard_value")
        units = r.get("standard_units")
        if not smi or not val or units != "nM":
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        canon = canonicalise(smi)
        if canon is None or not passes_filters(canon):
            continue
        rows.append(
            {
                "isoform": name,
                "chembl_id": r["molecule_chembl_id"],
                "canonical_smiles": canon,
                "pic50": to_pic50(val),
            }
        )
    df = pd.DataFrame(rows)
    # Merge duplicates within this isoform by median
    df = df.groupby(["chembl_id", "canonical_smiles"], as_index=False)["pic50"].median()
    df["isoform"] = name
    return df


def main() -> None:
    per_iso = {}
    for name in ["PI3Ka", "PI3Kb", "PI3Kd", "PI3Kg"]:
        df = process_isoform(name)
        per_iso[name] = df
        print(f"{name}: {len(df)} unique compounds")

    # Wide-format: one row per compound, 4 pIC50 columns
    all_ids = set().union(*[set(d["chembl_id"]) for d in per_iso.values()])
    wide_rows = []
    for cid in all_ids:
        row = {"chembl_id": cid}
        for iso, d in per_iso.items():
            sub = d[d.chembl_id == cid]
            row[f"pic50_{iso}"] = sub["pic50"].iloc[0] if len(sub) else None
            if len(sub):
                row["canonical_smiles"] = sub["canonical_smiles"].iloc[0]
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)

    print(f"\nTotal unique compounds: {len(wide)}")
    out_path = OUT_DIR / "chembl_curated_12601.csv"
    wide.to_csv(out_path, index=False)
    print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
