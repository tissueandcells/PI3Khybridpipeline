#!/bin/bash
# run_all.sh — Equilibration stages NVT1 → NVT2 for CPD_0332 / PI3Kβ complex
#
# Prerequisites:
#   - GROMACS 2026.0 (compiled with GPU support)
#   - System pre-prepared: topol.top, complex.gro, posre files from
#     scripts/07_md_validation/prepare_system.sh
#
# Outputs:
#   ~/md_work/production/CPD_0332/nvt1.{gro,cpt,xtc,edr,log}
#   ~/md_work/production/CPD_0332/nvt2.{gro,cpt,xtc,edr,log}

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/md_work/production/CPD_0332}"
MDP_DIR="${MDP_DIR:-$(dirname $0)/../../configs/mdp}"
GMX="${GMX:-gmx}"

cd "$WORKDIR"

echo "=== Energy minimization ==="
$GMX grompp -f "$MDP_DIR/em.mdp" -c complex_solv_ions.gro -p topol.top -o em.tpr -maxwarn 1
$GMX mdrun -deffnm em -ntmpi 1 -ntomp 8

echo "=== NVT Stage 1: 100 ps, 1000 kJ/mol/nm² restraints ==="
$GMX grompp -f "$MDP_DIR/nvt1.mdp" -c em.gro -r em.gro -p topol.top -o nvt1.tpr -maxwarn 1
$GMX mdrun -deffnm nvt1 -ntmpi 1 -ntomp 8 -nb gpu -pme cpu

echo "=== NVT Stage 2: 100 ps, 500 kJ/mol/nm² backbone restraints ==="
$GMX grompp -f "$MDP_DIR/nvt2.mdp" -c nvt1.gro -r nvt1.gro -t nvt1.cpt -p topol.top -o nvt2.tpr -maxwarn 1
$GMX mdrun -deffnm nvt2 -ntmpi 1 -ntomp 8 -nb gpu -pme cpu

echo "✓ Equilibration stages NVT1-NVT2 complete."
echo "  Next: run_rest.sh for NPT1-NPT2 + triplicate production."
