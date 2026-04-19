"""
scaffold_split.py — Bemis–Murcko scaffold-based train/val/test split.

Produces data/curated/scaffold_splits.json with 80/10/10 partition.
Prevents scaffold leakage between partitions.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "curated"

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10  # rest is test


def scaffold(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc, canonical=True)
    except Exception:
        return ""


def main() -> None:
    df = pd.read_csv(DATA_DIR / "chembl_curated_12601.csv")
    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        sc = scaffold(row["canonical_smiles"])
        groups[sc].append(row["chembl_id"])

    rng = random.Random(SEED)
    scaffolds = sorted(groups.keys(), key=lambda s: len(groups[s]), reverse=True)

    # Greedy allocation: largest scaffolds first → training
    n_total = len(df)
    n_train, n_val = int(n_total * TRAIN_FRAC), int(n_total * VAL_FRAC)
    train_ids, val_ids, test_ids = [], [], []
    for sc in scaffolds:
        if len(train_ids) < n_train:
            train_ids.extend(groups[sc])
        elif len(val_ids) < n_val:
            val_ids.extend(groups[sc])
        else:
            test_ids.extend(groups[sc])

    splits = {
        "seed": SEED,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
    }
    out_path = DATA_DIR / "scaffold_splits.json"
    out_path.write_text(json.dumps(splits, indent=2))
    print(f"✓ train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
