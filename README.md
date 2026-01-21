# DeepMD

DeepMD is a lightweight command-line pipeline for running OpenMM-based molecular dynamics workflows and training ensemble models from generated trajectories/logs. It provides stages for CMD runs, equilibration preparation/production, dataset windowing, and model training, all driven by a unified simulation configuration file.

## Features

- **Unified simulation config** with validation and support for YAML/JSON/TOML inputs.
- **CMD, equilibration prep, and production** stages for OpenMM-based workflows.
- **Dataset builder** to window and normalize trajectory logs for ML training.
- **Ensemble training** workflow with configurable model hyperparameters.
- **End-to-end pipeline** to run all steps in order with one command.

## Installation

This project is designed to be run as a local Python package. Install dependencies as needed (e.g., OpenMM, TensorFlow, NumPy, Pandas, PyYAML/TOML parser).

```bash
pip install -r requirements.txt
```

## Configuration

Simulation settings are defined in a config file and validated before runs. The configuration supports YAML (`.yaml/.yml`), JSON (`.json`), or TOML (`.toml`) formats. The configuration includes:

- Input topology/coordinate files (`parmFile`, `crdFile`).
- Simulation settings (`simType`, `nbCutoff`, `temperature`).
- Stage lengths and restart frequencies.
- Output directory, precision, and reproducibility options.

## CLI Usage

The CLI exposes multiple subcommands. The examples below assume the CLI entrypoint is available as `python -m paddle.cli` (adjust as needed for your environment).

### Run CMD stage

```bash
python -m paddle.cli cmd --config config.yaml --out out_cmd
```

### 5 ns CUDA test runs (explicit + implicit)

The CMD stage uses a 2 fs timestep (`dt_ps=0.002`), so **5 ns = 2,500,000 steps**. For a CUDA-backed OpenMM run, set the platform and CUDA parameters in your config, then run the CLI twice (explicit + implicit). Example configs:

**Explicit solvent (CUDA, 5 ns)**

```yaml
parmFile: topology/protein_solvated.parm7
crdFile: topology/protein_solvated.rst7
simType: explicit
nbCutoff: 10.0
temperature: 300.0
ntcmd: 2500000
cmdRestartFreq: 1000
platform: CUDA
precision: mixed
cuda_device_index: 0
cuda_precision: mixed
require_gpu: true
outdir: out_cmd_explicit_5ns
```

**Implicit solvent (CUDA, 5 ns)**

```yaml
parmFile: topology/protein_solvated.parm7
crdFile: topology/protein_solvated.rst7
simType: protein.implicit
temperature: 300.0
ntcmd: 2500000
cmdRestartFreq: 1000
platform: CUDA
precision: mixed
cuda_device_index: 0
cuda_precision: mixed
require_gpu: true
outdir: out_cmd_implicit_5ns
```

Run both 5 ns tests from the CLI:

```bash
python -m paddle.cli cmd --config config-explicit-5ns.yaml --out out_cmd_explicit_5ns
python -m paddle.cli cmd --config config-implicit-5ns.yaml --out out_cmd_implicit_5ns
```

### Run equilibration prep

```bash
python -m paddle.cli prep --config config.yaml --out out_prep
```

### Build dataset from prep logs

```bash
python -m paddle.cli data \
  --prep out_prep/prep \
  --out out_data \
  --window 128 \
  --stride 4 \
  --features Etot_kJ,Edih_kJ,T_K \
  --target Etot_kJ
```

### Train ensemble model

```bash
python -m paddle.cli train \
  --data out_data/windows.npz \
  --splits out_data/splits.json \
  --out out_models/run1
```

### Run equilibration + production

```bash
python -m paddle.cli equil_prod --config config.yaml --out out_prod
```

### Run the full pipeline

```bash
python -m paddle.cli pipeline --config config.yaml --out out_pipeline
```

## Outputs

- **Simulation logs and trajectories** stored in the configured output directory.
- **Prepared datasets** (`windows.npz`, `splits.json`) for ML training.
- **Trained ensemble models** stored in the provided output path.
