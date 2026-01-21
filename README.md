# DeepMD

Closed-loop Gaussian accelerated MD workflow with ML-assisted control.

## Project summary

Repository: `DeepMD`. This document summarizes the pipeline and implemented
functionality for reviewers.

## Quickstart

<!-- BEGIN GENERATED QUICKSTART -->
Install the package with your preferred environment manager. Ensure that Python can import
the package and that all simulation dependencies are available.

CLI entry point: run via `python cli.py`.

Minimal pipeline invocation (full workflow):

```bash
python cli.py pipeline --config config.yaml --out outdir
```

Generate example configuration YAMLs (writes explicit/implicit configs):

```bash
python cli.py make_configs --out configs
```

Example config files produced:

- `configs/config-explicit-cmd5ns-equil5ns-prod5ns.yml`
- `configs/config-implicit-cmd5ns-equil5ns-prod5ns.yml`

Create a working config by copying one of the generated files (for example,
`configs/config-explicit-cmd5ns-equil5ns-prod5ns.yml`) to `config.yml` and editing paths, GPU settings,
and run lengths as needed before invoking the pipeline.

### CLI commands

- `bench_alanine` — Generate alanine dipeptide benchmarks with tleap
- `cmd` — Run CMD stage
- `data` — Build dataset from prep logs
- `equil_prod` — Run equilibration + production
- `make_configs` — Generate example YAML configs for explicit/implicit CMD runs
- `pipeline` — Run full pipeline
- `prep` — Run equilibration-prep stage
- `train` — Train ensemble model

### Helpful CLI examples

Run just the CMD stage:

```bash
python cli.py cmd --config config.yml --out out_cmd
```

Run equilibration-prep and then build a training dataset:

```bash
python cli.py prep --config config.yml --out out_prep
python cli.py data --prep out_prep/prep --out out_data
```

Train an ensemble model from a prepared dataset:

```bash
python cli.py train --data out_data/windows.npz --splits out_data/splits.json --out out_models
```

Run equilibration + production only:

```bash
python cli.py equil_prod --config config.yml --out out_prod
```

Generate alanine dipeptide benchmarks:

```bash
python cli.py bench_alanine --out benchmarks/alanine
```
<!-- END GENERATED QUICKSTART -->

### Ablation comparison (not part of the pipeline)

The ablation plot is generated from the bias-plan logs after you complete two
pipeline runs (a controlled run and a baseline run). Run the plot script directly:

```bash
python scripts/plot_ablation_comparison.py \
  --controlled out_controlled \
  --baseline out_baseline \
  --out ablation_comparison.svg
```

## Pipeline stages

The pipeline is organized into stage modules. Each module lists its purpose and public entry
points for reviewers.

### cmd

Conventional MD (CMD) stage for PADDLE.

Key functions/classes:
- `run_cmd(cfg: SimulationConfig)`

### equil_prep

Equilibration preparation (collect energy series).

Key functions/classes:
- `run_equil_prep(cfg: SimulationConfig)`

### equil_prod

Equilibration (dual-boost) + Production.

Key functions/classes:
- `run_equil_and_prod(cfg: SimulationConfig)`

### pmf

Estimate 1D PMF from an energy-like series (demo).

Key functions/classes:
- `pmf_from_series(series: np.ndarray, bins: int = 50)`
- `save_pmf_json(path: str | Path, x: np.ndarray, pmf: np.ndarray)`

### report

Logging utilities for PADDLE.

Key functions/classes:
- `timestamp()`
- `ensure_dir(p: Path)`
- `class CSVLogger`
- `write_json(data: Mapping[str, object], path: Path)`
- `append_metrics(metrics: Mapping[str, object], outdir: Path, name: str = 'metrics.jsonl')`
- `write_run_manifest(outdir: Path, info: Mapping[str, object])`

### restart

Read/write restart metadata for PADDLE dual-boost runs.

Key functions/classes:
- `class RestartRecord`
- `class RestartFormatError`
- `read_restart(path: str | Path, k0_bounds: Tuple[float, float] = DEFAULT_K0_BOUNDS)`
- `write_restart(path: str | Path, rec: RestartRecord, k0_bounds: Tuple[float, float] =
  DEFAULT_K0_BOUNDS)`
- `backup(path: str | Path, backup_dir: str | Path)`
- `record_to_boost_params(rec: RestartRecord)`
- `validate_against_state(rec: RestartRecord, expected_steps: Optional[int] = None)`
- `normalize_restart_record(rec: RestartRecord, k0_bounds: Tuple[float, float] =
  DEFAULT_K0_BOUNDS)`

## Closed-loop controller

The controller implements a deterministic, rule-based policy that adapts GaMD bias
parameters using monitored simulation metrics. It is designed to maintain Gaussianity of
boosted potential energies while exploring conformational space responsibly.

Observation signals:
- Gaussianity metrics: skewness, excess kurtosis, and tail risk.
- Exploration score when available in per-cycle metrics.
- Latent diagnostics when a latent-space model is present.

Decision policy:
- Deterministic policy in `policy.py` that combines confidence, exploration, and uncertainty
  signals.
- Multi-objective alpha selection with optional change-point penalty.

Actuation:
- Updates to integrator parameters (k0D, k0P, refED/refEP bounds) per cycle.

Safety controls:
- Freeze criteria driven by Gaussianity and change-point detection.
- Uncertainty-aware damping to reduce aggressive updates.
- Hysteresis via confidence histories and damped update magnitudes.
- Multi-objective alpha blending for stability and exploration balance.

## ML model component

The ML component trains an ensemble model to predict observables used by the controller. An
ensemble refers to multiple models trained to quantify epistemic uncertainty, not multiple
simulation replicas. Conformal uncertainty calibration, when present, is summarized in the
model report output.

Model outputs:
- `model_summary.json` capturing ensemble statistics, uncertainty, and calibration data.

## Reweighting diagnostics

Reweighting diagnostics compute effective sample size (ESS), entropy-based measures, and
related convergence indicators when boost potential samples are available. The diagnostics
are integrated into per-cycle metrics and bias plans.

## Outputs and artifacts

- `bias_plan_cycle_*.json`: Per-cycle bias plan and controller state. Major keys include:
  controller, cycle, metrics, params; params keys: VmaxD, VmaxP, VminD, VminP, k0D, k0P,
  refED, refED_factor, refEP, refEP_factor; metrics keys: cycle_stats Detected in code scan.
- `metrics.jsonl / metrics.json`: Cycle metrics logs aggregated over the run. Detected in
  code scan.
- `model_summary.json`: Model ensemble summary and uncertainty statistics. Detected in code
  scan.
- `latent_pca.json`: Latent diagnostics used for Gaussianity analysis. Detected in code
  scan.
- `gamd-restart.dat`: GaMD restart checkpoint for continuing simulations. Detected in code
  scan.
- `controller_diagnostics.svg`: Closed-loop control diagnostics plots. Detected in code
  scan.
- `reweighting_diagnostics.svg`: Reweighting diagnostics plots. Detected in code scan.
- `ablation_comparison.svg`: Ablation comparison plot (only when running ablation scripts).
  Detected in code scan.
- `adaptive_stop.json`: Adaptive stopping decision record. Detected in code scan.

## Configuration reference

Configuration fields are parsed from `SimulationConfig`. Defaults and type annotations are
listed where available.

### system / I/O
- `parmFile: str` = `'topology/protein_solvated.parm7'`
- `crdFile: str` = `'topology/protein_solvated.rst7'`
- `simType: str` = `'explicit'`
- `safe_mode: bool` = `False`
- `platform: str` = `'CUDA'`
- `precision: str` = `'mixed'`
- `require_gpu: bool` = `False`
- `cuda_device_index: int` = `0`
- `cuda_precision: str` = `'mixed'`
- `deterministic_forces: bool` = `False`
- `seed: int` = `2025`
- `outdir: str` = `'out'`
- `compress_logs: bool` = `True`
- `notes: Optional[str]` = `None`

### MD / integrator
- `nbCutoff: float` = `10.0`
- `temperature: float` = `300.0`
- `dt: Optional[float]` = `None`
- `ntcmd: int` = `10000000`
- `cmdRestartFreq: int` = `100`
- `ncycebprepstart: int` = `0`
- `ncycebprepend: int` = `1`
- `ntebpreppercyc: int` = `2500000`
- `ebprepRestartFreq: int` = `100`
- `ncycebstart: int` = `0`
- `ncycebend: int` = `3`
- `ntebpercyc: int` = `2500000`
- `ebRestartFreq: int` = `100`
- `ncycprodstart: int` = `0`
- `ncycprodend: int` = `4`
- `ntprodpercyc: int` = `250000000`
- `prodRestartFreq: int` = `500`
- `refEP_factor: float` = `0.05`
- `refED_factor: float` = `0.05`
- `k0_initial: float` = `0.5`
- `k0_min: float` = `0.1`
- `k0_max: float` = `0.9`

### controller thresholds
- `gaussian_skew_good: float` = `0.2`
- `gaussian_excess_kurtosis_good: float` = `0.2`
- `gaussian_excess_kurtosis_high: float` = `1.0`
- `gaussian_tail_risk_good: float` = `0.01`
- `gaussian_skew_freeze: float` = `0.6`
- `gaussian_excess_kurtosis_freeze: float` = `2.0`
- `gaussian_tail_risk_freeze: float` = `0.05`
- `policy_damp_min: float` = `0.05`
- `policy_damp_max: float` = `1.0`
- `controller_enabled: bool` = `True`

### uncertainty calibration
- `uncertainty_ref: float` = `0.2`
- `uncertainty_damp_power: float` = `1.0`

### plotting
- No fields detected.

### stopping criteria
- No fields detected.

### other
- No fields detected.

## Enhancement features

- adaptive stop: Detected in code scan.
- auto plotting: Detected in code scan.
- change-point detection: Detected in code scan.
- closed-loop control: Detected in code scan.
- conformal uncertainty: Detected in code scan.
- freeze/damping: Detected in code scan.
- hysteresis: Detected in code scan.
- latent diagnostics: Detected in code scan.
- multi-objective control: Detected in code scan.
- reweighting diagnostics: Detected in code scan.

## Reproducibility

Runs are deterministic when seeds are fixed and deterministic force settings are enabled.
Environment manifests and JSON logs capture configuration, metrics, and controller
decisions, allowing reviewers to reproduce the pipeline and audit each control decision.
