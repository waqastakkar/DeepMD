# Pipeline Architecture

This document describes the current DeepMD pipeline layout, the CLI entry points, and the file-level artifacts produced at each stage.

## Top-level CLI

DeepMD is driven by a single CLI entry point:

```bash
python cli.py <command> [options]
```

The main commands are:

- `cmd` — Conventional MD stage
- `prep` — Equilibration-prep (energy series collection)
- `data` — Feature window generation from prep logs
- `train` — ML ensemble training on prep-derived windows
- `equil_prod` — GaMD equilibration + production
- `pipeline` — End-to-end CMD → prep → data → train → GaMD
- `make_configs` — Generate example configs
- `bench_alanine` — Generate alanine dipeptide benchmark inputs

See `src/paddle/cli.py` for the canonical argument list and defaults.

## Stage-by-stage architecture

### 1) MD setup

**Purpose:** load Amber inputs, select OpenMM platform, configure PME/GBn2 settings, and map force groups for energy decomposition.

**Key operations:**
- Explicit solvent uses PME with HBond constraints; implicit uses GBn2.
- Force-group mapping assigns bond/angle/dihedral/nonbonded contributions.

**Inputs:**
- `parmFile`, `crdFile` (Amber `parm7` + `rst7`)
- `simType` (`protein.explicit`, `protein.implicit`, `RNA.implicit`)
- `platform`, `precision`, `cuda_device_index`

**Outputs:** initialized OpenMM `Simulation` ready for integration.

### 2) CMD (Conventional MD)

**Purpose:** minimize, heat, and equilibrate density in NVT/NPT (explicit), then run conventional MD.

**Inputs:**
- MD configuration (time step, temperature, reporting interval)

**Outputs:**
- `cmd.dcd`, `cmd-state.log`, `md.log`
- `cmd.rst` checkpoint
- `prep_minimize.json`, `prep_heating.json`, `prep_density.json`

### 3) Equilibration-prep

**Purpose:** run a short conventional MD to collect energy components and thermodynamic statistics for feature extraction.

**Inputs:**
- CMD checkpoint (`cmd.rst`)
- reporting interval and energy decomposition settings

**Outputs:**
- `prep/equilprep-cycle*.csv(.gz)`
- `prep/metrics.jsonl` (per-cycle summaries)

### 4) Feature extraction + dataset construction

**Purpose:** transform prep logs into windowed ML-ready features.

**Inputs:**
- prep CSV logs

**Outputs:**
- `windows.npz` (feature windows + targets)
- `splits.json`, `stats.json`

### 5) GaMD equilibration

**Purpose:** configure dual-boost GaMD integrator, adaptively update boost parameters, and monitor Gaussianity/reweighting metrics.

**Inputs:**
- Equilibrated state (CMD checkpoint or fresh prep stages)
- GaMD control parameters (`k0_*`, `refE*_factor`)

**Outputs:**
- `equil/equil-cycle*.dcd`, `equil/equil-cycle*.log`
- `equil/gamd-diagnostics-cycle*.csv` (energies, ΔV, stats)
- `equil/gamd-ml-cycle*.csv` (ML-friendly feature stream)
- `bias_plan_cycle_*.json`, `gamd-restart.dat`

### 6) GaMD production

**Purpose:** run dual-boost GaMD production and emit diagnostics.

**Outputs:**
- `prod/prod-cycle*.dcd`, `prod/prod-cycle*.log`
- `prod/gamd-diagnostics-cycle*.csv`, `prod/gamd-ml-cycle*.csv`
- `metrics.jsonl`

### 7) GNN modeling + saliency analysis

**Purpose:** train a residue-level GNN and compute saliency maps.

**Inputs:**
- Residue-level `.npz` with node features and positions

**Outputs:**
- `gamd_gnn_model.keras`
- `importance.csv` + JSON metadata
- Optional edge-weight exports (network analysis)

## Directory layout

A typical GaMD output tree looks like:

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

For GaMD theory details, see `docs/gamd.md`. For the GNN workflow, see `docs/gnn.md`.
