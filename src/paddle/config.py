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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    allowed = {"explicit", "protein.implicit", "RNA.implicit"}
    if s not in allowed:
        raise ValueError(f"simType must be one of {sorted(allowed)}, got: {s}")
    return s


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
    parmFile: str = "topology/protein_solvated.parm7"  # was Name.top
    crdFile:  str = "topology/protein_solvated.rst7"  # was Name.crd

    # Explicit solvent mode (this flips engine.py to PME + HBonds + barostat)
    simType: str = "explicit"                           

    # Explicit-water friendly cutoff
    nbCutoff: float = 10.0                             

    temperature: float = 300.0

    ntcmd: int = 10_000_000
    cmdRestartFreq: int = 100

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
    ntprodpercyc: int = 250_000_000
    prodRestartFreq: int = 500

    refEP_factor: float = 0.05
    refED_factor: float = 0.05
    k0_initial: float = 0.5
    k0_min: float = 0.1
    k0_max: float = 0.9

    gaussian_skew_good: float = 0.2
    gaussian_excess_kurtosis_good: float = 0.2
    gaussian_excess_kurtosis_high: float = 1.0
    gaussian_tail_risk_good: float = 0.01

    platform: str = "CUDA"
    precision: str = "mixed"
    require_gpu: bool = False
    cuda_device_index: int = 0
    cuda_precision: str = "mixed"
    deterministic_forces: bool = False  # set True if you need bitwise reproducibility

    seed: int = 2025
    outdir: str = "out"
    compress_logs: bool = True
    notes: Optional[str] = None


    def validate(self) -> None:
        _exists_if_set(Path(self.parmFile), "parmFile")
        _exists_if_set(Path(self.crdFile), "crdFile")
        self.simType = _coerce_simtype(self.simType)

        _assert_range("nbCutoff", self.nbCutoff, 0.0, 50.0)
        _assert_range("temperature", self.temperature, 1.0, 2000.0)

        for name in [
            "ntcmd", "cmdRestartFreq", "ntebpreppercyc", "ebprepRestartFreq",
            "ntebpercyc", "ebRestartFreq", "ntprodpercyc", "prodRestartFreq",
            "ncycebprepend", "ncycebend", "ncycprodend"
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

        if self.precision not in {"single", "mixed", "double"}:
            raise ValueError("precision must be one of: single|mixed|double")
        if self.cuda_precision not in {"single", "mixed", "double"}:
            raise ValueError("cuda_precision must be one of: single|mixed|double")
        if self.require_gpu and self.platform != "CUDA":
            raise ValueError("require_gpu is true but platform is not CUDA")

        Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def to_yaml(self) -> str:
        if yaml is None:
            raise RuntimeError("pyyaml is not installed. Install with: pip install pyyaml")
        return yaml.safe_dump(self.as_dict(), sort_keys=False)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimulationConfig":
        cfg = SimulationConfig(**d)
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
    for field in dc.fields(SimulationConfig):
        key = field.name
        if hasattr(ns, key):
            val = getattr(ns, key)
            if val is not None:
                setattr(cfg, key, val)
    if getattr(ns, "deterministic_forces", False):
        cfg.deterministic_forces = True
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
