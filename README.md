Pre-registered Molecular Dynamics Validation Falsifies a Rigid-Docking Polar Triad and Reveals a Conserved HRD-Aspartate Anchor in Isoform-Selective PI3K Inhibitor Discovery

[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)


---

## Overview

This repository contains the complete code, curated datasets, trained model weights, docking outputs, and molecular dynamics inputs for a validation-first hybrid computational pipeline that predicts isoform-selective inhibitors for the four Class I PI3K isoforms (α, β, δ, γ).

**Key contributions:**

1. A **multi-task graph neural network** (MT-GNN) trained on 12,601 curated ChEMBL compounds with scaffold-split validation, providing an inverse-scaling data-efficiency gain (ΔR² = +0.31 to +0.52, p < 10⁻¹² for all four isoforms).
2. A **dual positive–negative control validation** strategy with graded specificity layers (9 positive controls + 22 decoys across 3 layers) that diagnoses task-specific ML failure modes — in particular, a PI3Kδ training-set bias and PIKK-family cross-reactivity — before committing to a pipeline architecture.
3. A **complementary bias framework** (ML: δ-bias; single-structure docking: γ-bias; ensemble docking: α/β-confusion) that justifies the two-stage pipeline design on empirical rather than a priori grounds.
4. **Twelve top candidate compounds** (three per isoform) with predicted selectivity, validated binding-mode pharmacophores, and favourable ADMET profiles — including a novel polar-augmented β-selective binding mode hypothesis subjected to triplicate 100 ns molecular dynamics validation.

---

## Pipeline architecture

```
12,601 ChEMBL compounds
        │
        ▼  [ML stage: MT-GNN v2 potency pre-filter]
8,533 candidates (max pIC₅₀ > 7)
        │
        ▼  [Tier 1: Lipinski + Veber + PAINS + Brenk + QED ≥ 0.3]
638 drug-like compounds
        │
        ▼  [Tier 2: ADMET-AI — 6 hard filters]
494 ADMET-passed compounds
        │
        ▼  [Structure stage: ensemble docking into 17 holo PI3K structures]
10,824 successful dockings (99.8%)
        │
        ▼  [Pharmacophore-informed ranking by ΔΔG and overlap]
12 top candidates (3 per isoform)
        │
        ▼  [MD validation of β-selective polar-augmented binding hypothesis]
CPD_0332 triplicate 100 ns production
```

**Reduction factor:** ~1,050× (12,601 → 12)

---

## Repository layout

```
PI3K_hybrid_pipeline/
├── README.md                          # this file
├── LICENSE                            # code license (MIT)
├── LICENSE_DATA                       # data license (CC-BY 4.0)
├── CITATION.cff                       # machine-readable citation metadata
├── requirements.txt                   # pinned Python dependencies
├── environment.yml                    # conda environment specification
├── Makefile                           # one-command pipeline entry points
│
├── data/
│   ├── raw/                           # ChEMBL 34 raw downloads
│   ├── curated/                       # 12,601-compound curated dataset + scaffold splits
│   ├── controls/                      # dual-control cohorts (9 positive + 22 decoys)
│   ├── docking/                       # 10,846-run ensemble docking outputs
│   ├── pharmacophore/                 # pharmacophore features, overlap matrices
│   └── md/                            # MD system files (CPD_0332 / PI3Kβ)
│
├── models/
│   ├── mtgnn_v1.pt                    # baseline MT-GNN checkpoint
│   ├── mtgnn_v2.pt                    # selectivity-aware MT-GNN checkpoint
│   └── single_task/                   # four single-task ablation checkpoints
│
├── scripts/
│   ├── 01_data_curation/              # ChEMBL fetch, canonicalisation, scaffold split
│   ├── 02_mtgnn_training/             # v1 and v2 training, hyperparameter configs
│   ├── 03_dual_controls/              # positive + negative control evaluation
│   ├── 04_filtering/                  # Tier 1 drug-likeness + Tier 2 ADMET
│   ├── 05_ensemble_docking/           # 10,846-run docking pipeline with checkpointing
│   ├── 06_pharmacophore/              # pharmacophore extraction, overlap, robustness
│   └── 07_md_validation/              # MD preparation, run_all.sh, run_rest.sh
│

```

---

## Installation

### Conda environment (recommended)

```bash
git clone https://github.com/tissueandcells/PI3K_hybrid_pipeline.git
cd PI3K_hybrid_pipeline
conda env create -f environment.yml
conda activate pi3k_pipeline
```

### Pip

```bash
pip install -r requirements.txt
```

### External tools

The following non-pip tools must be installed separately:

| Tool | Version | Purpose |
|---|---|---|
| AutoDock Vina | 1.2.3 | Molecular docking |
| OpenBabel | 3.1.1 | PDBQT conversion |
| GROMACS | 2026.0 | Molecular dynamics |
| PDBFixer | 1.11 | PDB preprocessing |
| ACPYPE | 2023.10.27 | Ligand parameterisation |
| ChimeraX | ≥1.8 | Visualisation (Figure 9 panels) |

See [`docs/installation.md`](docs/installation.md) for platform-specific notes, GPU driver requirements, and known conflicts.

---

## Quick start

Reproduce the entire pipeline from raw ChEMBL downloads to the 12-candidate final list:

```bash
make all
```

Or step by step:

```bash
make data          # 1. Fetch + curate ChEMBL data (~30 min)
make train         # 2. Train MT-GNN v1 and v2 (~2 h on RTX 5060 Ti)
make controls      # 3. Run dual positive + negative control evaluation (~5 min)
make filter        # 4. Apply Tier 1 + Tier 2 filters (~10 min)
make dock          # 5. Full ensemble docking campaign (~24 h)
make pharma        # 6. Pharmacophore analysis + robustness checks (~15 min)
make figures       # 7. Regenerate all publication figures
```

The molecular dynamics validation for CPD_0332 requires manual launch owing to its ~8-day wall-clock time:

```bash
cd scripts/07_md_validation
bash run_all.sh     # equilibration stages NVT1 → NVT2
bash run_rest.sh    # NPT1 → NPT2 + triplicate 100 ns production
```

See [`docs/md_protocol.md`](docs/md_protocol.md) for the complete MD reproduction protocol.

---

## Trained model weights

Pre-trained MT-GNN checkpoints are provided in `models/`:

| File | Description | Reported in |
|---|---|---|
| `mtgnn_v1.pt` | Baseline MT-GNN (22-d atom features, MSE loss) | Tables 2–3, Figure 4 |
| `mtgnn_v2.pt` | Selectivity-aware MT-GNN (32-d features, inverse-frequency-weighted MSE + pairwise ranking loss weight 0.3) | Tables 10, Figure 5, downstream pipeline |
| `single_task/mtgnn_st_{α,β,δ,γ}.pt` | Four single-task ablation checkpoints | Table 3 (ST columns), Figure 4 |

Loading example:

```python
import torch
from scripts.02_mtgnn_training.model import MTGNN

model = MTGNN(in_dim=32, hidden_dim=256)
model.load_state_dict(torch.load("models/mtgnn_v2.pt", map_location="cpu"))
model.eval()
```

---



## Data availability

All derived datasets are deposited under `data/` in machine-readable CSV/JSON format:

- `data/curated/chembl_curated_12601.csv` — final curated dataset
- `data/curated/scaffold_splits.json` — train/val/test partitions (Bemis–Murcko scaffolds)
- `data/controls/positive_controls.csv` — 25 PI3K inhibitors + ChEMBL ground-truth pIC₅₀
- `data/controls/negative_controls.csv` — 22-compound layered decoy set
- `data/docking/campaign_638_results.csv` — 10,846-run ensemble docking output
- `data/pharmacophore/supp_S3_full_features.csv` — 175 pharmacophore features (12 hits + 4 refs)
- `data/md/CPD_0332_PI3Kβ/` — starting coordinates, topology, MDP files for MD replication

Raw ChEMBL API dumps are in `data/raw/` (reproducible from the manifest in `scripts/01_data_curation/chembl_targets.yaml`).

---



