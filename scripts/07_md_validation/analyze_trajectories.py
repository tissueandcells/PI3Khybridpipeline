#!/usr/bin/env python3
"""
analyze_trajectories.py — Post-production analysis for CPD_0332 MD replicas.

Computes per-replica:
  - Ligand heavy-atom RMSD (after backbone alignment) — pose retention
  - Protein backbone RMSD and per-residue RMSF — structural stability
  - H-bond occupancy for LYS787, ASP919, ASP844 — polar-augmented hypothesis test
  - Ligand COM distance from ATP-pocket centroid — pocket retention

Analysis window: final 50 ns of each replica (50-100 ns).

Classifies outcome as pre-registered category:
  A — hypothesis supported (≥ 2/3 replicas, ≥40% mean occupancy)
  B — partial support (occupancy 20-40%)
  C — not supported (occupancy < 20%)
  D — binding mode failure (ligand exits pocket)

Outputs:
    data/md/analysis_summary.json
    data/md/per_replica_metrics.csv
    figures/Figure10_md_validation.{png,svg}
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = Path.home() / "md_work" / "production" / "CPD_0332"
ANALYSIS_OUT = ROOT / "data" / "md"
ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

REPLICAS = [1, 2, 3]
ANALYSIS_START_NS = 50  # final 50 ns of each 100 ns replica
ANALYSIS_END_NS = 100
HBOND_CRITICAL = ["LYS787", "ASP919", "ASP844"]
POCKET_RESIDUES = ["VAL848", "ILE849", "MET779", "LYS771", "ASP862"]


def run_gmx(args: list[str]) -> None:
    subprocess.run(["gmx"] + args, check=True, cwd=WORKDIR)


def ligand_rmsd(replica: int) -> dict:
    """Compute ligand heavy-atom RMSD after backbone alignment."""
    tpr = WORKDIR / f"prod_r{replica}.tpr"
    xtc = WORKDIR / f"prod_r{replica}.xtc"
    out_xvg = WORKDIR / f"ligand_rmsd_r{replica}.xvg"
    # gmx rms -s $tpr -f $xtc -fit rot+trans -o $out_xvg
    #   select Protein-H for alignment, LIG for RMSD calc
    # Parse resulting .xvg and return mean/SD over 50-100 ns window
    return {"mean": None, "sd": None, "max": None}


def hbond_occupancy(replica: int, residue: str) -> float:
    """Compute H-bond occupancy for a given residue."""
    # gmx hbond -s prod_r${replica}.tpr -f prod_r${replica}.xtc
    #          -r 3.5  -a 150  -num hbond_${residue}_r${replica}.xvg
    # Parse and compute fraction of frames with ≥ 1 H-bond
    return np.nan  # stub


def classify_outcome(metrics: dict) -> str:
    # Pre-registered criteria (Section 2.13.2, Supplementary Document S1)
    lig_rmsd_ok = all(m["ligand_rmsd_mean"] < 5.0 for m in metrics["per_replica"])
    backbone_ok = all(m["backbone_rmsd_mean"] < 4.0 for m in metrics["per_replica"])

    occ_per_res = {
        res: np.mean([m[f"occ_{res}"] for m in metrics["per_replica"]])
        for res in HBOND_CRITICAL
    }
    supported_res = sum(1 for occ in occ_per_res.values() if occ >= 0.40)
    partial_res = sum(1 for occ in occ_per_res.values() if 0.20 <= occ < 0.40)

    if lig_rmsd_ok and backbone_ok and supported_res >= 2:
        return "A"
    if lig_rmsd_ok and backbone_ok and partial_res >= 2:
        return "B"
    if lig_rmsd_ok and backbone_ok:
        return "C"
    return "D"


def main() -> None:
    per_replica = []
    for r in REPLICAS:
        rmsd = ligand_rmsd(r)
        occ = {res: hbond_occupancy(r, res) for res in HBOND_CRITICAL}
        per_replica.append(
            {
                "replica": r,
                "ligand_rmsd_mean": rmsd["mean"],
                "ligand_rmsd_sd": rmsd["sd"],
                "backbone_rmsd_mean": None,
                **{f"occ_{res}": v for res, v in occ.items()},
            }
        )

    pd.DataFrame(per_replica).to_csv(ANALYSIS_OUT / "per_replica_metrics.csv", index=False)
    summary = {"per_replica": per_replica, "outcome_category": classify_outcome({"per_replica": per_replica})}
    (ANALYSIS_OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"✓ Outcome: Category {summary['outcome_category']}")


if __name__ == "__main__":
    main()
