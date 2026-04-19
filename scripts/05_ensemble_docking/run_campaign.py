#!/usr/bin/env python3
"""
run_campaign.py — Full ensemble docking campaign over 494 ADMET-passed compounds
                  into 17 holo PI3K crystal structures (8,398 runs).

Features:
  - Checkpoint-based persistence: writes to disk every 100 completed runs,
    resumes without loss of completed work on restart.
  - Ligand-flexibility-aware exhaustiveness: 16 (≤15 torsions) / 32 (>15 torsions).
  - 600-second wall-clock timeout per run.
  - Tracks success rate and flags failures for inspection.

Usage:
    python run_campaign.py --resume   # resume from last checkpoint
    python run_campaign.py --fresh    # start from scratch (overwrites checkpoint)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
STRUCTURES_DIR = ROOT / "data" / "docking" / "structures"
LIGANDS_DIR = ROOT / "data" / "docking" / "ligands"
RESULTS_DIR = ROOT / "data" / "docking" / "campaign_638"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

STRUCTURES = {
    "PI3Ka": ["4JPS", "4OVV_excluded", "3ZIM", "4L1B"],
    "PI3Kb": ["2Y3A", "2WAB", "AF2_P42338"],
    "PI3Kd": ["4XE0", "5AE8", "5IS5", "6PYR", "6MVB", "6G6W"],
    "PI3Kg": ["3L08", "3QJZ", "3QKO", "4ANV", "4V0H"],
}
VINA_BIN = "vina"
EXHAUSTIVENESS_LOW = 16
EXHAUSTIVENESS_HIGH = 32
TORSION_THRESHOLD = 15
TIMEOUT_S = 600


def load_checkpoint() -> set[tuple[str, str]]:
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text())
        return {tuple(pair) for pair in data.get("completed", [])}
    return set()


def save_checkpoint(completed: set[tuple[str, str]]) -> None:
    CHECKPOINT.write_text(json.dumps({"completed": list(completed)}, indent=2))


def count_torsions(ligand_pdbqt: Path) -> int:
    txt = ligand_pdbqt.read_text()
    return txt.count("ACTIVE") + txt.count("BRANCH")


def dock_one(lig: Path, rec: Path, out_path: Path, exhaust: int) -> bool:
    cmd = [
        VINA_BIN,
        "--receptor", str(rec),
        "--ligand", str(lig),
        "--out", str(out_path),
        "--exhaustiveness", str(exhaust),
        "--num_modes", "9",
        "--seed", "42",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_S)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    completed = load_checkpoint() if not args.fresh else set()

    ligands = sorted(LIGANDS_DIR.glob("*.pdbqt"))
    print(f"→ {len(ligands)} ligands × {sum(len(v) for v in STRUCTURES.values())} structures")
    print(f"  Resuming: {len(completed)} completed runs")

    failures = []
    tic = time.time()
    for lig in tqdm(ligands, desc="Ligands"):
        cpd_id = lig.stem
        tors = count_torsions(lig)
        exhaust = EXHAUSTIVENESS_HIGH if tors > TORSION_THRESHOLD else EXHAUSTIVENESS_LOW
        for iso, structs in STRUCTURES.items():
            for pdb in structs:
                if "excluded" in pdb:
                    continue
                pair = (cpd_id, pdb)
                if pair in completed:
                    continue
                rec = STRUCTURES_DIR / f"{pdb}.pdbqt"
                out = RESULTS_DIR / "poses" / f"{cpd_id}__{pdb}.pdbqt"
                out.parent.mkdir(parents=True, exist_ok=True)
                ok = dock_one(lig, rec, out, exhaust)
                if ok:
                    completed.add(pair)
                else:
                    failures.append(pair)
                if len(completed) % 100 == 0:
                    save_checkpoint(completed)

    save_checkpoint(completed)
    print(f"\n✓ Completed {len(completed)} runs in {(time.time()-tic)/3600:.1f} h")
    print(f"  Failures: {len(failures)} ({100*len(failures)/max(len(completed)+len(failures),1):.2f}%)")
    if failures:
        (RESULTS_DIR / "failures.json").write_text(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
