# Configuration Reference

DeepMD uses a unified `SimulationConfig` parsed from YAML, JSON, or TOML. Paths are resolved relative to the config file location. All numeric values use OpenMM conventions (ps, K, atm) unless noted.

## Supported formats

- `.yml` / `.yaml` (recommended)
- `.json`
- `.toml`

## Minimal example

```yaml
parmFile: /path/to/complex.parm7
crdFile: /path/to/complex.rst7
simType: protein.explicit
platform: CUDA
require_gpu: true
cuda_device_index: 0
cuda_precision: mixed
precision: mixed
outdir: out_gamd
```

## Core system and I/O

| Key | Purpose |
| --- | --- |
| `parmFile` | Amber `parm7` topology file |
| `crdFile` | Amber `rst7` coordinates |
| `simType` | `protein.explicit`, `protein.implicit`, `RNA.implicit` |
| `outdir` | Output directory |
| `safe_mode` | Strict validation of time-step settings |

## MD and integrator timing

| Key | Purpose |
| --- | --- |
| `dt` | Time step in ps (default 0.002) |
| `cmd_ns` | Total CMD length (ns) |
| `equil_ns_per_cycle` | GaMD equilibration length per cycle (ns) |
| `prod_ns_per_cycle` | GaMD production length per cycle (ns) |
| `cmdRestartFreq` | CMD reporting interval (steps) |
| `ebRestartFreq` | GaMD equilibration reporting interval |
| `prodRestartFreq` | GaMD production reporting interval |

## Minimization, heating, and density equilibration

| Key | Purpose |
| --- | --- |
| `do_minimize` | Toggle minimization |
| `minimize_max_iter` | Maximum minimization iterations |
| `do_heating` | Toggle temperature ramp |
| `heat_t_start`, `heat_t_end` | Temperature schedule (K) |
| `heat_ns` | Heating duration (ns) |
| `do_density_equil` | Toggle NPT density equilibration |
| `density_ns` | Density equilibration duration (ns) |
| `pressure_atm` | Target pressure (atm) |
| `barostat_interval` | Barostat update interval |

## GaMD parameters

| Key | Purpose |
| --- | --- |
| `k0_initial` | Initial k0 value for GaMD |
| `k0_min`, `k0_max` | Bounds on k0 |
| `refED_factor`, `refEP_factor` | Clamp factors for reference energies |
| `sigma0D`, `sigma0P` | Optional GaMD sigma targets |
| `gamd_diag_enabled` | Enable GaMD diagnostics CSVs |

## Gaussianity / control thresholds

| Key | Purpose |
| --- | --- |
| `gaussian_skew_good` | Target skewness threshold |
| `gaussian_excess_kurtosis_good` | Target kurtosis threshold |
| `gaussian_tail_risk_good` | Tail-risk threshold |
| `gaussian_*_freeze` | Freeze thresholds for controller |
| `controller_enabled` | Enable/disable adaptive GaMD control |

## GPU and OpenMM platform

| Key | Purpose |
| --- | --- |
| `platform` | `CUDA`, `OpenCL`, or `CPU` |
| `precision` | OpenMM precision (single/mixed/double) |
| `cuda_precision` | CUDA precision override |
| `cuda_device_index` | GPU index |
| `require_gpu` | Enforce CUDA platform |
| `deterministic_forces` | Deterministic force calculations |

## Logging and reproducibility

| Key | Purpose |
| --- | --- |
| `compress_logs` | Gzip compress CSV logs |
| `seed` | RNG seed for reproducibility |
| `notes` | Optional run notes |

For the full schema, consult `src/paddle/config.py` and the generated example configs from `python cli.py make_configs`.
