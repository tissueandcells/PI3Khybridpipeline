"""
fetch_chembl.py — Retrieve PI3K bioactivity data from ChEMBL 34.

Fetches IC50 records for all four Class I PI3K isoforms and writes raw JSON
dumps to data/raw/. Subsequent canonicalisation and curation is performed
by curate_dataset.py.

Usage:
    python fetch_chembl.py
"""
import json
import time
from pathlib import Path

from chembl_webresource_client.new_client import new_client

# Class I PI3K catalytic subunits in ChEMBL 34
TARGETS = {
    "PI3Ka": "CHEMBL4005",
    "PI3Kb": "CHEMBL3145",
    "PI3Kd": "CHEMBL3130",
    "PI3Kg": "CHEMBL3267",
}

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_target(target_name: str, target_id: str) -> list[dict]:
    """Fetch all IC50 records for a given ChEMBL target id."""
    print(f"→ Fetching {target_name} ({target_id}) ...")
    activities = new_client.activity
    records = activities.filter(
        target_chembl_id=target_id,
        standard_type="IC50",
        standard_relation="=",
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_relation",
        "standard_type",
        "target_chembl_id",
        "assay_chembl_id",
        "document_chembl_id",
    )
    out = list(records)
    print(f"   retrieved {len(out)} records")
    return out


def main() -> None:
    for name, cid in TARGETS.items():
        records = fetch_target(name, cid)
        out_path = OUT_DIR / f"chembl34_{name}_raw.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"   wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
        time.sleep(2)  # rate-limit courtesy

    print("✓ All four isoforms fetched.")


if __name__ == "__main__":
    main()
