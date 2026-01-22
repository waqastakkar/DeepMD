# DeepMD: GaMD–GNN Molecular Dynamics Pipeline

A GPU-accelerated, CLI-driven workflow for explicit-solvent OpenMM simulations with Gaussian accelerated MD (GaMD), energy decomposition, ML-ready feature extraction, graph neural network (GNN) training, and residue-level saliency mapping.

# Overview

DeepMD orchestrates a complete pipeline from classical molecular dynamics (MD) through GaMD-accelerated sampling, feature extraction, GNN modeling, and saliency analysis. The workflow follows:

MD → GaMD → Feature extraction → GNN training → Saliency mapping.

Each stage is modular, reproducible, and designed for high-throughput GPU execution using OpenMM, with downstream ML components implemented in TensorFlow.

# Key Features

- OpenMM explicit-solvent MD (Amber prmtop/inpcrd; recommended ff19SB + OPC workflows)
- GaMD acceleration with dual-boost integrators
- Energy decomposition via force-group mapping (bond/angle/dihedral/nonbonded)
- ML-ready feature extraction from equilibration logs
- GNN-based residue-level saliency mapping
- GPU acceleration with CUDA and CPU fallback
- Reproducibility tooling and CI-aware outputs

# Installation

## Conda environment (recommended)

```bash
conda create -n deepmd python=3.10 -y
conda activate deepmd
conda install -c conda-forge openmm cudatoolkit -y
pip install -e .
```

If you prefer pip-only installation (CPU-only OpenMM), install the requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

## CUDA requirements

- NVIDIA GPU with a CUDA-capable driver.
- OpenMM built with CUDA support (conda-forge `openmm` package supports CUDA). 
- To target a specific device, set `cuda_device_index` in the YAML config and/or use `CUDA_VISIBLE_DEVICES`.

## CPU fallback

Set `platform: CPU` in your YAML config (and `require_gpu: false`). The pipeline will run on CPU with reduced performance.

# Quick Start

Below is a minimal end-to-end example using the built-in alanine dipeptide benchmark and the provided GNN example dataset.

## 1) Build a test system (ff19SB + OPC)

```bash
python cli.py bench_alanine --out benchmarks/alanine
```

The command creates explicit/implicit Amber inputs and a ready-to-run `config.yml` in each output directory.

## 2) Run a short CMD stage

```bash
python cli.py cmd --config benchmarks/alanine/explicit/config.yml --out out_cmd
```

## 3) Run GaMD equilibration + production

```bash
python cli.py equil_prod --config benchmarks/alanine/explicit/config.yml --out out_gamd
```

## 4) Generate ML features from equilibration logs

```bash
python cli.py prep --config benchmarks/alanine/explicit/config.yml --out out_prep
python cli.py data --prep out_prep/prep --out out_data
```

## 5) Train the GNN and generate saliency scores

```bash
python scripts/gnn_example.py --out examples
python scripts/gnn_pipeline.py --npz examples/alanine_gamd.npz --out out_gnn
```

## 6) Summarize saliency

```bash
python scripts/gnn_visualize.py --importance out_gnn/importance.csv --out out_gnn/saliency_report.txt
```

# Pipeline Architecture

For additional diagrams and rationale, see `docs/architecture.md`.

## 1) MD Setup
**Inputs:** Amber `parm7` + `rst7`, YAML configuration.
**Outputs:** Initialized OpenMM `Simulation` with PME (explicit) or GBn2 (implicit).

## 2) CMD (Conventional MD)
**Inputs:** MD setup + integrator.
**Outputs:** `cmd.dcd`, `cmd-state.log`, `md.log`, `cmd.rst`, `prep_*.json` summaries.

## 3) Heating
**Inputs:** minimized coordinates.
**Outputs:** temperature ramp metadata in `prep_heating.json`.

## 4) Density Equilibration (NPT)
**Inputs:** explicit-solvent system with Monte Carlo barostat.
**Outputs:** `prep_density.json` summary and stabilized density.

## 5) GaMD Equilibration
**Inputs:** CMD checkpoint or equilibrated state.
**Outputs:** `equil/*.dcd`, `equil/*.log`, `gamd-diagnostics-cycle*.csv`, `gamd-ml-cycle*.csv`, `bias_plan_cycle_*.json`, `gamd-restart.dat`.

## 6) GaMD Production
**Inputs:** GaMD restart state.
**Outputs:** `prod/*.dcd`, `prod/*.log`, `gamd-diagnostics-cycle*.csv`, `gamd-ml-cycle*.csv`, `metrics.jsonl`.

## 7) Feature Extraction
**Inputs:** `prep/equilprep-cycle*.csv(.gz)` logs.
**Outputs:** `windows.npz`, `splits.json`, `stats.json` for ML training.

## 8) Dataset Construction
**Inputs:** Equilibration features + windowing configuration.
**Outputs:** normalized feature windows and train/val/test splits.

## 9) GNN Training
**Inputs:** Residue-level `.npz` with node features and positions (see `docs/gnn.md`).
**Outputs:** `gamd_gnn_model.keras`, saliency CSV/JSON.

## 10) Saliency Mapping
**Inputs:** trained GNN model + residue features.
**Outputs:** `importance.csv`, `importance.csv.json`, optional edge-weight exports.

### Directory layout (typical)

```
out_gamd/
├── cmd.dcd
├── cmd.rst
├── cmd-state.log
├── md.log
├── equil/
│   ├── equil-cycle00.dcd
│   ├── equil-cycle00.log
│   ├── gamd-diagnostics-cycle00.csv
│   └── gamd-ml-cycle00.csv
├── prod/
│   ├── prod-cycle00.dcd
│   ├── prod-cycle00.log
│   ├── gamd-diagnostics-cycle00.csv
│   └── gamd-ml-cycle00.csv
├── bias_plan_cycle_0.json
├── gamd-restart.dat
├── metrics.jsonl
└── prep_density.json
```

# Configuration System

Configurations are YAML/JSON/TOML files parsed into `SimulationConfig`. See `docs/config.md` for a full reference.

Key YAML sections include:

- **System I/O:** `parmFile`, `crdFile`, `simType`, `outdir`
- **MD timing:** `dt`, `cmd_ns`, `equil_ns_per_cycle`, `prod_ns_per_cycle`, reporting intervals
- **Thermostat/barostat:** `temperature`, `pressure_atm`, `barostat_interval`
- **GaMD parameters:** `refEP_factor`, `refED_factor`, `k0_initial`, `k0_min`, `k0_max`
- **GPU options:** `platform`, `precision`, `cuda_device_index`, `cuda_precision`, `require_gpu`
- **Control + diagnostics:** Gaussianity thresholds, reweighting diagnostics, closed-loop controller toggles

Example explicit ff19SB + OPC GaMD configs (dual boost and dihedral-only):

```yaml
# Dual-boost GaMD (explicit ff19SB + OPC)
parmFile: topology/complex_ff19sb_opc.parm7
crdFile: topology/complex_ff19sb_opc.rst7
simType: protein.explicit
temperature: 300
dt: 0.002
controller_enabled: true
debug_disable_gamd: false
deltaV_std_max: 10.0
k_min: 0.05
k_max: 0.9
sigma0D: 6.0
sigma0P: 10.0
gamd_ramp_ns: 0.5
deltaV_abs_max: 2000.0
safe_mode: false  # if true and dt is omitted, dt is overridden to 0.001 ps
```

```yaml
# Dihedral-only GaMD (explicit ff19SB + OPC)
parmFile: topology/complex_ff19sb_opc.parm7
crdFile: topology/complex_ff19sb_opc.rst7
simType: protein.explicit
temperature: 300
dihedral_only: true
controller_enabled: true
debug_disable_gamd: false
deltaV_std_max: 6.0
k_min: 0.05
k_max: 0.9
sigma0D: 6.0
sigma0P: 10.0
gamd_ramp_ns: 0.5
deltaV_abs_max: 2000.0
```

# GaMD Theory (Concise but correct)

GaMD adds a harmonic boost potential to smooth the energy landscape when the system’s potential energy falls below a threshold energy **E**. The boost is:

ΔV = ½ k (E − V)^2 for V < E, and ΔV = 0 for V ≥ E.

The workflow supports:

- **Dihedral boost** (dihedral energy only)
- **Total boost** (total potential energy)
- **Dual boost** (simultaneous dihedral + total boosts)

The threshold **E** and force constant **k** are adaptively chosen from running energy statistics, with safeguards to preserve near-Gaussian boost distributions for accurate reweighting. Reweighting is supported through cumulant/entropy diagnostics to recover unbiased free-energy estimates.

# Feature Extraction

Feature extraction operates on equilibration-prep logs and yields ML-ready time windows. Reported quantities include:

- **Energy decomposition:** bond, angle, dihedral, nonbonded, total potential energy (force-group mapping)
- **Thermodynamic observables:** temperature, density
- **Derived statistics:** mean/std/min/max per window

Outputs are serialized to `windows.npz` (features + targets), `splits.json` (train/val/test indices), and `stats.json` (normalization metadata).

# GNN Model

The GNN pipeline constructs residue-level graphs per frame using contact edges, optional physical interactions (H-bonds, salt bridges, covariance), and distance/direction edge features. The model combines:

- **Graph encoder:** SE(3)-aware message passing + multi-head graph attention
- **Temporal encoder:** convolutional layers + multi-head self-attention across trajectory windows
- **Multi-task heads:** ΔV regression, state classification, RMSD, Rg, and latent projection

Training uses sliding windows and is driven by the CLI in `scripts/gnn_pipeline.py`.

# Saliency Mapping

Residue-level saliency is computed using gradient-based methods and attention-derived attributions:

- **Gradient attribution:** saliency from ∂output/∂node features
- **Integrated gradients:** baseline-to-input path integral
- **Attention rollout:** aggregation across GAT layers
- **GraphCAM:** gradient-weighted node embedding projection

Outputs include `importance.csv` and a JSON metadata sidecar, plus optional edge-weight exports for network visualization.

# Example Commands

## Full CLI command list

```bash
python cli.py cmd --config config.yml --out out_cmd
python cli.py prep --config config.yml --out out_prep
python cli.py data --prep out_prep/prep --out out_data
python cli.py train --data out_data/windows.npz --splits out_data/splits.json --out out_models
python cli.py equil_prod --config config.yml --out out_gamd
python cli.py pipeline --config config.yml --out out_pipeline
python cli.py make_configs --out configs
python cli.py bench_alanine --out benchmarks/alanine
```

## Typical workflows

**Short test run**
```bash
python cli.py make_configs --out configs
python cli.py cmd --config configs/config-explicit-cmd5ns-equil5ns-prod5ns.yml --out out_cmd
```

**Production GaMD run**
```bash
python cli.py equil_prod --config config.yml --out out_gamd
```

**Feature generation**
```bash
python cli.py prep --config config.yml --out out_prep
python cli.py data --prep out_prep/prep --out out_data
```

**GNN training**
```bash
python scripts/gnn_pipeline.py --npz path/to/residue_features.npz --out out_gnn --epochs 25 --sequence 8 --batch 4
```

**Saliency visualization**
```bash
python scripts/gnn_visualize.py --importance out_gnn/importance.csv --out out_gnn/saliency_report.txt
```

# Outputs

- **Trajectories:** `cmd.dcd`, `equil/*.dcd`, `prod/*.dcd`
- **Logs:** `cmd-state.log`, `md.log`, `equil/*.log`, `prod/*.log`
- **GaMD diagnostics:** `gamd-diagnostics-cycle*.csv`, `gamd-ml-cycle*.csv`, `bias_plan_cycle_*.json`, `gamd-restart.dat`
- **Feature tables:** `prep/equilprep-cycle*.csv(.gz)`, `windows.npz`, `stats.json`
- **Models:** `gamd_gnn_model.keras`, ensemble model outputs, checkpoints
- **Figures:** controller and reweighting diagnostics (optional)
- **Saliency maps:** `importance.csv` + JSON metadata

# Performance Guidelines

- **Expected throughput:** system-dependent; explicit solvent with GaMD typically achieves tens of ns/day on modern GPUs.
- **GPU vs CPU:** CUDA provides order-of-magnitude speedups for MD and GNN training.
- **I/O optimization:** increase report intervals for high-throughput runs; compress CSV logs with `compress_logs: true`.
- **Reporting frequencies:** balance temporal resolution with storage (e.g., `cmdRestartFreq`, `ebRestartFreq`, `prodRestartFreq`).

# Troubleshooting

- **GPU not detected:** verify `platform: CUDA`, OpenMM CUDA install, and `CUDA_VISIBLE_DEVICES`.
- **Barostat not active:** ensure explicit solvent (`simType: protein.explicit`) and `barostat_interval` > 0.
- **Edih_kJ or E_dihedral_kJ zero:** confirm torsion force groups exist in topology and force-group mapping is enabled.
- **CI failure:** ensure OpenMM/TensorFlow versions match your environment manifest.
- **Missing dependencies:** `pip install -r requirements.txt` and verify `pyyaml` for YAML configs.

# Citation

If you use DeepMD in your work, please cite:

> DeepMD: Closed-loop GaMD with ML-assisted analysis. (2025).

**BibTeX stub**

```bibtex
@misc{deepmd2025,
  title        = {DeepMD: Closed-loop GaMD with ML-assisted analysis},
  author       = {DeepMD Development Team},
  year         = {2025},
  howpublished = {\url{https://github.com/your-org/DeepMD}},
}
```

Additional documentation:

- `docs/architecture.md`
- `docs/gamd.md`
- `docs/gnn.md`
- `docs/config.md`
