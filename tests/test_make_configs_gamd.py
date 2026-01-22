from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.cli import cmd_make_configs  # noqa: E402


def _run_make_configs(tmp_path: Path) -> None:
    ns = SimpleNamespace(
        out=str(tmp_path),
        cmd_ns=5.0,
        equil_ns_per_cycle=5.0,
        prod_ns_per_cycle=5.0,
        heat_ns=0.2,
        density_ns=0.5,
        explicit_config=None,
        implicit_config=None,
    )
    cmd_make_configs(ns)


def _assert_gamd_keys(data: dict) -> None:
    assert data["validate_config"] is True
    assert data["controller_enabled"] is True
    assert data["gamd_boost_mode"] in {"dual", "dihedral"}
    for key in ("deltaV_std_max", "k_min", "k_max", "sigma0D", "sigma0P"):
        assert math.isfinite(float(data[key]))


def test_make_configs_gamd_outputs(tmp_path: Path) -> None:
    _run_make_configs(tmp_path)
    for name in ("gamd_dual_opc_ff19sb.yaml", "gamd_dihedral_safe.yaml"):
        data = yaml.safe_load((tmp_path / name).read_text())
        _assert_gamd_keys(data)
