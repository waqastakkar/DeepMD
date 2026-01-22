"""
config.py — Unified configuration for paddle simulations
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
import os
import platform
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # noqa: N816

try:
    import tomllib  # type: ignore
except Exception:
    tomllib = None  # type: ignore
    try:
        import tomli as tomllib  # type: ignore
    except Exception:
        tomllib = None  # type: ignore


def _coerce_simtype(s: str) -> str:
    s = s.strip()
    if s == "explicit":
        return "protein.explicit"
    allowed = {"protein.explicit", "protein.implicit", "RNA.implicit"}
    if s not in allowed:
        raise ValueError(f"simType must be one of {sorted(allowed)}, got: {s}")
    return s


def _rst7_has_box_vectors(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    if len(lines) < 2:
        return False
    header = lines[1].split()
    if not header:
        return False
    try:
        natoms = int(float(header[0]))
    except Exception:
        return False
    coord_tokens = " ".join(lines[2:]).split()
    if natoms <= 0:
        return False
    expected_coords = 3 * natoms
    return len(coord_tokens) >= expected_coords + 6


def _infer_simtype(parm_file: str, crd_file: str) -> Optional[str]:
    parm_name = Path(parm_file).name.lower() if parm_file else ""
    if "solv" in parm_name:
        return "protein.explicit"
    crd_path = Path(crd_file) if crd_file else None
    if crd_path and crd_path.exists() and _rst7_has_box_vectors(crd_path):
        return "protein.explicit"
    if parm_file or crd_file:
        return "protein.implicit"
    return None


def is_explicit_simtype(sim_type: str) -> bool:
    return sim_type == "explicit" or sim_type.endswith(".explicit")


def ns_to_steps(ns: float, dt_ps: float) -> int:
    return int(round((ns * 1000.0) / dt_ps))


def steps_to_ns(steps: int, dt_ps: float) -> float:
    return (steps * dt_ps) / 1000.0


def _sanity_check_timesteps(cfg: "SimulationConfig") -> None:
    cfg.check_timestep_consistency()


def _assert_range(name: str, val: float, lo: float | None = None, hi: float | None = None) -> None:
    if lo is not None and val < lo:
        raise ValueError(f"{name} must be >= {lo}, got {val}")
    if hi is not None and val > hi:
        raise ValueError(f"{name} must be <= {hi}, got {val}")


def _exists_if_set(path: Path, name: str) -> None:
    if path and not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _resolve_config_paths(data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = dict(data)
    for key in ("parmFile", "crdFile", "outdir"):
        value = resolved.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            resolved[key] = str(base_dir / path)
    return resolved


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass


@dataclass
class SimulationConfig:
    # Use the solvated Amber files you built in tleap
    parmFile: str = "topology/complex.parm7"  # was Name.top
    crdFile:  str = "topology/complex.rst7"  # was Name.crd

    # Explicit solvent mode (this flips engine.py to PME + HBonds + barostat)
    simType: str = "protein.explicit"

    # Explicit-water friendly cutoff
    nbCutoff: float = 10.0                             

    temperature: float = 300.0
    dt: Optional[float] = 0.002
    safe_mode: bool = False
    validate_config: bool = False
    cmd_ns: float = 5.0
    equil_ns_per_cycle: float = 5.0
    prod_ns_per_cycle: float = 5.0
    feature_columns: List[str] = dc.field(default_factory=lambda: [
        "E_potential_kJ",
        "E_bond_kJ",
        "E_angle_kJ",
        "E_dihedral_kJ",
        "E_nonbonded_kJ",
        "T_K",
    ])

    ntcmd: int = 2_500_000
    cmdRestartFreq: int = 100

    do_minimize: bool = True
    minimize_max_iter: int = 5000
    minimize_tolerance_kj_per_mol: float = 10.0

    do_heating: bool = True
    heat_t_start: float = 0.0
    heat_t_end: float = 300.0
    heat_ns: float = 0.2
    ntheat: int = 100_000
    heat_report_freq: int = 1000

    do_density_equil: bool = True
    density_ns: float = 0.5
    ntdensity: int = 250_000
    pressure_atm: float = 1.0
    barostat_interval: int = 25
    density_report_freq: int = 1000

    ncycebprepstart: int = 0
    ncycebprepend: int = 1
    ntebpreppercyc: int = 2_500_000
    ebprepRestartFreq: int = 100

    ncycebstart: int = 0
    ncycebend: int = 3
    ntebpercyc: int = 2_500_000
    ebRestartFreq: int = 100

    ncycprodstart: int = 0
    ncycprodend: int = 4
    ntprodpercyc: int = 2_500_000
    prodRestartFreq: int = 500

    refEP_factor: float = 0.05
    refED_factor: float = 0.05
    k0_initial: float = 0.5
    k0_min: float = 0.1
    k0_max: float = 0.9
    k_min: Optional[float] = None
    k_max: Optional[float] = None
    sigma0D: Optional[float] = None
    sigma0P: Optional[float] = None

    gaussian_skew_good: float = 0.2
    gaussian_excess_kurtosis_good: float = 0.2
    gaussian_excess_kurtosis_high: float = 1.0
    gaussian_tail_risk_good: float = 0.01

    # --- NEW: closed-loop control safety + uncertainty scaling ---
    gaussian_skew_freeze: float = 0.6
    gaussian_excess_kurtosis_freeze: float = 2.0
    gaussian_tail_risk_freeze: float = 0.05
    deltaV_std_max: Optional[float] = None
    deltaV_damp_factor: float = 0.5

    policy_damp_min: float = 0.05
    policy_damp_max: float = 1.0

    uncertainty_ref: float = 0.2
    uncertainty_damp_power: float = 1.0
    controller_enabled: bool = True

    platform: str = "CUDA"
    precision: str = "mixed"
    require_gpu: bool = False
    cuda_device_index: Optional[int] = None
    cuda_precision: str = "mixed"
    deterministic_forces: bool = False  # set True if you need bitwise reproducibility
    ewaldErrorTolerance: float = 5e-4
    useDispersionCorrection: bool = True
    rigidWater: bool = True
    gamd_diag_enabled: bool = True

    seed: int = 2025
    outdir: str = "out"
    compress_logs: bool = True
    notes: Optional[str] = None


    def validate(self) -> None:
        _exists_if_set(Path(self.parmFile), "parmFile")
        _exists_if_set(Path(self.crdFile), "crdFile")
        self.simType = _coerce_simtype(self.simType)
        inferred = _infer_simtype(self.parmFile, self.crdFile)
        if inferred and self.simType != inferred:
            self.simType = inferred

        _assert_range("nbCutoff", self.nbCutoff, 0.0, 50.0)
        _assert_range("temperature", self.temperature, 1.0, 2000.0)
        if self.dt is None:
            self.dt = 0.002
        _assert_range("dt", self.dt, 1e-9)
        _assert_range("cmd_ns", self.cmd_ns, 1e-12)
        _assert_range("equil_ns_per_cycle", self.equil_ns_per_cycle, 1e-12)
        _assert_range("prod_ns_per_cycle", self.prod_ns_per_cycle, 1e-12)
        if self.k_min is not None and self.k_max is not None:
            if self.k_min > self.k_max:
                raise ValueError(f"k_min must be <= k_max, got {(self.k_min, self.k_max)}")
        if self.sigma0D is not None:
            _assert_range("sigma0D", self.sigma0D, 0.0)
        if self.sigma0P is not None:
            _assert_range("sigma0P", self.sigma0P, 0.0)
        _assert_range("minimize_tolerance_kj_per_mol", self.minimize_tolerance_kj_per_mol, 0.0)
        _assert_range("heat_ns", self.heat_ns, 0.0)
        _assert_range("heat_t_start", self.heat_t_start, 0.0, 5000.0)
        _assert_range("heat_t_end", self.heat_t_end, 0.0, 5000.0)
        _assert_range("density_ns", self.density_ns, 0.0)
        _assert_range("pressure_atm", self.pressure_atm, 0.0, 10000.0)

        for name in [
            "ntcmd", "cmdRestartFreq", "ntebpreppercyc", "ebprepRestartFreq",
            "ntebpercyc", "ebRestartFreq", "ntprodpercyc", "prodRestartFreq",
            "ncycebprepend", "ncycebend", "ncycprodend",
            "minimize_max_iter", "ntheat", "heat_report_freq",
            "ntdensity", "barostat_interval", "density_report_freq",
        ]:
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val}")

        _assert_range("refEP_factor", self.refEP_factor, 0.0, 1.0)
        _assert_range("refED_factor", self.refED_factor, 0.0, 1.0)
        _assert_range("k0_initial", self.k0_initial, 0.0, 1.0)
        _assert_range("k0_min", self.k0_min, 0.0, 1.0)
        _assert_range("k0_max", self.k0_max, 0.0, 1.0)
        if self.k0_min > self.k0_max:
            raise ValueError("k0_min must be <= k0_max")

        _assert_range("gaussian_skew_good", self.gaussian_skew_good, 0.0, 10.0)
        _assert_range("gaussian_excess_kurtosis_good", self.gaussian_excess_kurtosis_good, 0.0, 10.0)
        _assert_range("gaussian_excess_kurtosis_high", self.gaussian_excess_kurtosis_high, 0.0, 10.0)
        _assert_range("gaussian_tail_risk_good", self.gaussian_tail_risk_good, 0.0, 1.0)

        _assert_range("gaussian_skew_freeze", self.gaussian_skew_freeze, 0.0, 10.0)
        _assert_range(
            "gaussian_excess_kurtosis_freeze",
            self.gaussian_excess_kurtosis_freeze,
            0.0,
            10.0,
        )
        _assert_range("gaussian_tail_risk_freeze", self.gaussian_tail_risk_freeze, 0.0, 1.0)
        if self.deltaV_std_max is not None:
            _assert_range("deltaV_std_max", self.deltaV_std_max, 0.0)
        _assert_range("deltaV_damp_factor", self.deltaV_damp_factor, 0.0, 1.0)

        _assert_range("policy_damp_min", self.policy_damp_min, 0.0, 1.0)
        _assert_range("policy_damp_max", self.policy_damp_max, 0.0, 1.0)
        if self.policy_damp_min > self.policy_damp_max:
            raise ValueError("policy_damp_min must be <= policy_damp_max")

        _assert_range("uncertainty_ref", self.uncertainty_ref, 0.0, 1e9)
        _assert_range("uncertainty_damp_power", self.uncertainty_damp_power, 0.0, 10.0)
        if not isinstance(self.controller_enabled, bool):
            raise ValueError("controller_enabled must be a bool")
        for name in ("do_minimize", "do_heating", "do_density_equil"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

        if self.precision not in {"single", "mixed", "double"}:
            raise ValueError("precision must be one of: single|mixed|double")
        if self.cuda_precision not in {"single", "mixed", "double"}:
            raise ValueError("cuda_precision must be one of: single|mixed|double")
        if self.cuda_device_index is not None:
            if not isinstance(self.cuda_device_index, int) or self.cuda_device_index < 0:
                raise ValueError("cuda_device_index must be a non-negative integer if set")
        _assert_range("ewaldErrorTolerance", self.ewaldErrorTolerance, 1e-8, 1e-2)
        for name in ("useDispersionCorrection", "rigidWater"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.require_gpu and self.platform != "CUDA":
            raise ValueError("require_gpu is true but platform is not CUDA")

        if not isinstance(self.feature_columns, list) or not self.feature_columns:
            raise ValueError("feature_columns must be a non-empty list of strings")
        if not all(isinstance(col, str) and col.strip() for col in self.feature_columns):
            raise ValueError("feature_columns must contain only non-empty strings")
        dupes = {col for col in self.feature_columns if self.feature_columns.count(col) > 1}
        if dupes:
            raise ValueError(f"feature_columns contains duplicates: {sorted(dupes)}")

        from paddle.policy import validate_config as validate_policy_config

        validate_policy_config(self)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if self.safe_mode or getattr(self, "validate_config", False):
            _sanity_check_timesteps(self)

    def reconcile_time_settings(self, *, input_keys: set[str]) -> None:
        if self.dt is None:
            self.dt = 0.002
        dt_ps = float(self.dt)

        def _check_inconsistency(name: str, expected: int, actual: int, ns_name: str) -> None:
            if expected != actual:
                msg = (
                    f"{name}={actual} is inconsistent with {ns_name}={getattr(self, ns_name)} "
                    f"and dt={dt_ps}; expected {expected}"
                )
                if self.safe_mode:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

        if "cmd_ns" in input_keys:
            expected = ns_to_steps(self.cmd_ns, dt_ps)
            if "ntcmd" in input_keys:
                _check_inconsistency("ntcmd", expected, self.ntcmd, "cmd_ns")
            self.ntcmd = expected
        elif "ntcmd" in input_keys:
            self.cmd_ns = steps_to_ns(self.ntcmd, dt_ps)

        if "equil_ns_per_cycle" in input_keys:
            expected = ns_to_steps(self.equil_ns_per_cycle, dt_ps)
            if "ntebpreppercyc" in input_keys:
                _check_inconsistency("ntebpreppercyc", expected, self.ntebpreppercyc, "equil_ns_per_cycle")
            if "ntebpercyc" in input_keys:
                _check_inconsistency("ntebpercyc", expected, self.ntebpercyc, "equil_ns_per_cycle")
            self.ntebpreppercyc = expected
            self.ntebpercyc = expected
        elif "ntebpercyc" in input_keys:
            self.equil_ns_per_cycle = steps_to_ns(self.ntebpercyc, dt_ps)
        elif "ntebpreppercyc" in input_keys:
            self.equil_ns_per_cycle = steps_to_ns(self.ntebpreppercyc, dt_ps)

        if "prod_ns_per_cycle" in input_keys:
            expected = ns_to_steps(self.prod_ns_per_cycle, dt_ps)
            if "ntprodpercyc" in input_keys:
                _check_inconsistency("ntprodpercyc", expected, self.ntprodpercyc, "prod_ns_per_cycle")
            self.ntprodpercyc = expected
        elif "ntprodpercyc" in input_keys:
            self.prod_ns_per_cycle = steps_to_ns(self.ntprodpercyc, dt_ps)

        if "heat_ns" in input_keys:
            expected = ns_to_steps(self.heat_ns, dt_ps)
            if "ntheat" in input_keys:
                _check_inconsistency("ntheat", expected, self.ntheat, "heat_ns")
            self.ntheat = expected
        elif "ntheat" in input_keys:
            self.heat_ns = steps_to_ns(self.ntheat, dt_ps)

        if "density_ns" in input_keys:
            expected = ns_to_steps(self.density_ns, dt_ps)
            if "ntdensity" in input_keys:
                _check_inconsistency("ntdensity", expected, self.ntdensity, "density_ns")
            self.ntdensity = expected
        elif "ntdensity" in input_keys:
            self.density_ns = steps_to_ns(self.ntdensity, dt_ps)

    def check_timestep_consistency(self) -> None:
        if self.dt is None:
            self.dt = 0.002
        dt_ps = float(self.dt)
        expected_cmd = ns_to_steps(self.cmd_ns, dt_ps)
        expected_equil = ns_to_steps(self.equil_ns_per_cycle, dt_ps)
        expected_prod = ns_to_steps(self.prod_ns_per_cycle, dt_ps)
        expected_heat = ns_to_steps(self.heat_ns, dt_ps)
        expected_density = ns_to_steps(self.density_ns, dt_ps)
        if self.ntcmd != expected_cmd:
            raise ValueError(f"ntcmd={self.ntcmd} does not match cmd_ns={self.cmd_ns} (expected {expected_cmd})")
        if self.ntebpreppercyc != expected_equil:
            raise ValueError(
                f"ntebpreppercyc={self.ntebpreppercyc} does not match equil_ns_per_cycle="
                f"{self.equil_ns_per_cycle} (expected {expected_equil})"
            )
        if self.ntebpercyc != expected_equil:
            raise ValueError(
                f"ntebpercyc={self.ntebpercyc} does not match equil_ns_per_cycle="
                f"{self.equil_ns_per_cycle} (expected {expected_equil})"
            )
        if self.ntprodpercyc != expected_prod:
            raise ValueError(
                f"ntprodpercyc={self.ntprodpercyc} does not match prod_ns_per_cycle="
                f"{self.prod_ns_per_cycle} (expected {expected_prod})"
            )
        if self.ntheat != expected_heat:
            raise ValueError(
                f"ntheat={self.ntheat} does not match heat_ns={self.heat_ns} (expected {expected_heat})"
            )
        if self.ntdensity != expected_density:
            raise ValueError(
                f"ntdensity={self.ntdensity} does not match density_ns={self.density_ns} (expected {expected_density})"
            )

    def as_dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def to_yaml(self) -> str:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed. Install with: pip install pyyaml")
        data = self.as_dict()
        data["dt"] = float(self.dt or 0.002)
        return yaml.safe_dump(data, sort_keys=False)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimulationConfig":
        cfg = SimulationConfig(**d)
        cfg.reconcile_time_settings(input_keys=set(d.keys()))
        cfg.validate()
        return cfg

    @staticmethod
    def _load_file(path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("pyyaml is required to read YAML: pip install pyyaml")
            return dict(yaml.safe_load(text) or {})
        if suffix == ".json":
            return dict(json.loads(text))
        if suffix == ".toml":
            if tomllib is None:
                raise RuntimeError("tomli/tomllib required to read TOML: pip install tomli")
            return dict(tomllib.loads(text))  # type: ignore
        raise ValueError(f"Unsupported config format: {suffix}")

    @staticmethod
    def from_file(path: str | Path) -> "SimulationConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = SimulationConfig._load_file(p)
        data = _resolve_config_paths(data, p.parent)
        return SimulationConfig.from_dict(data)


def _detect_openmm_version() -> Optional[str]:
    try:
        import openmm  # type: ignore
        return getattr(openmm, "__version__", "unknown")
    except Exception:
        return None


def _detect_tf_version() -> Optional[str]:
    try:
        import tensorflow as tf  # type: ignore
        return getattr(tf, "__version__", "unknown")
    except Exception:
        return None


def write_env_manifest(outdir: str) -> Path:
    manifest = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "openmm": _detect_openmm_version(),
        "tensorflow": _detect_tf_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "openmm_default_platform": os.environ.get("OPENMM_DEFAULT_PLATFORM"),
    }
    p = Path(outdir) / "env-manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DBMDX configuration utility")
    ap.add_argument("--config", type=str, help="Path to YAML/JSON/TOML config", default=None)
    ap.add_argument("--parmFile", type=str, default=None)
    ap.add_argument("--crdFile", type=str, default=None)
    ap.add_argument("--simType", type=str, default=None)
    ap.add_argument("--nbCutoff", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cmd_ns", type=float, default=None)
    ap.add_argument("--equil_ns_per_cycle", type=float, default=None)
    ap.add_argument("--prod_ns_per_cycle", type=float, default=None)
    ap.add_argument("--do_minimize", action="store_true", default=None)
    ap.add_argument("--do_heating", action="store_true", default=None)
    ap.add_argument("--do_density_equil", action="store_true", default=None)
    ap.add_argument("--minimize_max_iter", type=int, default=None)
    ap.add_argument("--minimize_tolerance_kj_per_mol", type=float, default=None)
    ap.add_argument("--heat_t_start", type=float, default=None)
    ap.add_argument("--heat_t_end", type=float, default=None)
    ap.add_argument("--heat_ns", type=float, default=None)
    ap.add_argument("--ntheat", type=int, default=None)
    ap.add_argument("--heat_report_freq", type=int, default=None)
    ap.add_argument("--density_ns", type=float, default=None)
    ap.add_argument("--ntdensity", type=int, default=None)
    ap.add_argument("--pressure_atm", type=float, default=None)
    ap.add_argument("--barostat_interval", type=int, default=None)
    ap.add_argument("--density_report_freq", type=int, default=None)
    ap.add_argument("--ntcmd", type=int, default=None)
    ap.add_argument("--cmdRestartFreq", type=int, default=None)
    ap.add_argument("--ncycebprepstart", type=int, default=None)
    ap.add_argument("--ncycebprepend", type=int, default=None)
    ap.add_argument("--ntebpreppercyc", type=int, default=None)
    ap.add_argument("--ebprepRestartFreq", type=int, default=None)
    ap.add_argument("--ncycebstart", type=int, default=None)
    ap.add_argument("--ncycebend", type=int, default=None)
    ap.add_argument("--ntebpercyc", type=int, default=None)
    ap.add_argument("--ebRestartFreq", type=int, default=None)
    ap.add_argument("--ncycprodstart", type=int, default=None)
    ap.add_argument("--ncycprodend", type=int, default=None)
    ap.add_argument("--ntprodpercyc", type=int, default=None)
    ap.add_argument("--prodRestartFreq", type=int, default=None)
    ap.add_argument("--refEP_factor", type=float, default=None)
    ap.add_argument("--refED_factor", type=float, default=None)
    ap.add_argument("--platform", type=str, default=None)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--require_gpu", action="store_true", default=None)
    ap.add_argument("--cuda_device_index", type=int, default=None)
    ap.add_argument("--cuda_precision", type=str, default=None)
    ap.add_argument("--deterministic_forces", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--notes", type=str, default=None)
    ap.add_argument("--echo", action="store_true", help="Print validated config and exit")
    ap.add_argument("--write-template", type=str, default=None, help="Write a template YAML/JSON/TOML file and exit")
    ap.add_argument("--write-json", type=str, default=None, help="Write current config to JSON and exit")
    ap.add_argument("--write-yaml", type=str, default=None, help="Write current config to YAML and exit")
    return ap


def _apply_overrides(cfg: SimulationConfig, ns: argparse.Namespace) -> SimulationConfig:
    overridden: set[str] = set()
    for field in dc.fields(SimulationConfig):
        key = field.name
        if hasattr(ns, key):
            val = getattr(ns, key)
            if val is not None:
                setattr(cfg, key, val)
                overridden.add(key)
    if getattr(ns, "deterministic_forces", False):
        cfg.deterministic_forces = True
        overridden.add("deterministic_forces")
    if overridden & {
        "cmd_ns",
        "equil_ns_per_cycle",
        "prod_ns_per_cycle",
        "ntcmd",
        "ntebpreppercyc",
        "ntebpercyc",
        "ntprodpercyc",
        "heat_ns",
        "ntheat",
        "density_ns",
        "ntdensity",
    }:
        cfg.reconcile_time_settings(input_keys=overridden)
    return cfg


def _write_template(path: Path) -> None:
    cfg = SimulationConfig()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed. Install with: pip install pyyaml")
        path.write_text(cfg.to_yaml(), encoding="utf-8")
    elif path.suffix.lower() == ".json":
        path.write_text(cfg.to_json(2), encoding="utf-8")
    elif path.suffix.lower() == ".toml":
        if tomllib is None:
            raise RuntimeError("tomli/tomllib is required for TOML. Install: pip install tomli")
        def to_toml(d: Dict[str, Any]) -> str:
            lines = []
            for k, v in d.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                elif v is None:
                    lines.append(f"{k} = null")
                else:
                    s = str(v).replace("\n", " ")
                    lines.append(f"{k} = \"{s}\"")
            return "\n".join(lines) + "\n"
        path.write_text(to_toml(cfg.as_dict()), encoding="utf-8")
    else:
        raise ValueError("Template suffix must be .yaml/.yml/.json/.toml")


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_arg_parser()
    ns = ap.parse_args(argv)

    if ns.write_template:
        _write_template(Path(ns.write_template))
        print(f"Wrote template: {ns.write_template}")
        return 0

    if ns.config:
        cfg = SimulationConfig.from_file(ns.config)
    else:
        cfg = SimulationConfig()

    cfg = _apply_overrides(cfg, ns)
    cfg.validate()

    set_global_seed(cfg.seed)
    man = write_env_manifest(cfg.outdir)

    if ns.echo:
        print("Configuration (validated):")
        print(json.dumps(cfg.as_dict(), indent=2))
        print(f"\nEnv manifest: {man}")

    if ns.write_json:
        Path(ns.write_json).write_text(cfg.to_json(2), encoding="utf-8")
        print(f"Wrote JSON: {ns.write_json}")

    if ns.write_yaml:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed. Install with: pip install pyyaml")
        Path(ns.write_yaml).write_text(cfg.to_yaml(), encoding="utf-8")
        print(f"Wrote YAML: {ns.write_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
