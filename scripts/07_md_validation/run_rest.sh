#!/bin/bash
# run_rest.sh — NPT1 → NPT2 → triplicate 100 ns production for CPD_0332 / PI3Kβ
#
# CRITICAL: NPT1 and NPT2 use refcoord_scaling = com (in npt1.mdp / npt2.mdp)
# which is REQUIRED for stable pressure coupling under position restraints.
# Omitting this directive causes rapid box volume collapse.
#
# Production uses HMR + 4 fs timestep + all-bonds LINCS (Hopkins et al., 2015).
#
# Three replicas with distinct velocity seeds (42, 123, 456):
#   - prod_r1.xtc (seed 42)
#   - prod_r2.xtc (seed 123)
#   - prod_r3.xtc (seed 456)
#
# Total wall-clock: ~8 days on RTX 5060 Ti 16 GB (~38 ns/day per replica).

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/md_work/production/CPD_0332}"
MDP_DIR="${MDP_DIR:-$(dirname $0)/../../configs/mdp}"
GMX="${GMX:-gmx}"

cd "$WORKDIR"

echo "=== NPT Stage 1: 500 ps with C-rescale + refcoord_scaling=com ==="
$GMX grompp -f "$MDP_DIR/npt1.mdp" -c nvt2.gro -r nvt2.gro -t nvt2.cpt -p topol.top -o npt1.tpr -maxwarn 1
$GMX mdrun -deffnm npt1 -ntmpi 1 -ntomp 8 -nb gpu -pme cpu

echo "=== NPT Stage 2: 1 ns gentle release, HMR + 4 fs + all-bonds ==="
$GMX grompp -f "$MDP_DIR/npt2.mdp" -c npt1.gro -t npt1.cpt -p topol.top -o npt2.tpr -maxwarn 1
$GMX mdrun -deffnm npt2 -ntmpi 1 -ntomp 8 -nb gpu -pme cpu

# Triplicate production with per-replica seeds
for REPLICA in 1 2 3; do
    case $REPLICA in
        1) SEED=42  ;;
        2) SEED=123 ;;
        3) SEED=456 ;;
    esac
    echo "=== Production Replica $REPLICA (seed=$SEED): 100 ns ==="

    # Regenerate velocities per replica by editing grompp input
    sed "s/gen-seed.*=.*/gen-seed = $SEED/" "$MDP_DIR/production.mdp" > prod_r${REPLICA}.mdp
    # Enable gen-vel for replica start
    sed -i 's/gen-vel.*=.*/gen-vel = yes/' prod_r${REPLICA}.mdp
    echo "gen-temp = 300" >> prod_r${REPLICA}.mdp

    $GMX grompp -f prod_r${REPLICA}.mdp -c npt2.gro -t npt2.cpt -p topol.top \
                -o prod_r${REPLICA}.tpr -maxwarn 1
    $GMX mdrun -deffnm prod_r${REPLICA} -ntmpi 1 -ntomp 8 -nb gpu -pme cpu -bonded gpu
done

echo "✓ Triplicate production complete."
echo "  Analysis: scripts/07_md_validation/analyze_trajectories.py"
