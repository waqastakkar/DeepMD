from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.config import SimulationConfig  # noqa: E402
from paddle.stages.equil_prod import _resolve_dt_ps  # noqa: E402


def _base_config(tmp_path: Path) -> dict:
    parm = tmp_path / "system.parm7"
    crd = tmp_path / "system.rst7"
    parm.write_text("parm")
    crd.write_text("crd")
    return {
        "parmFile": str(parm),
        "crdFile": str(crd),
        "outdir": str(tmp_path / "out"),
    }


def test_missing_gamd_keys_raise_value_error(tmp_path):
    cfg_data = _base_config(tmp_path)
    cfg_data.update({
        "controller_enabled": True,
        "debug_disable_gamd": False,
    })
    with pytest.raises(ValueError) as excinfo:
        SimulationConfig.from_dict(cfg_data)
    message = str(excinfo.value)
    for key in ("deltaV_std_max", "k_min", "k_max", "sigma0D", "sigma0P"):
        assert key in message


def test_safe_mode_dt_override_only_when_not_user_provided(tmp_path):
    cfg_data = _base_config(tmp_path)
    cfg_data.update({
        "safe_mode": True,
        "controller_enabled": False,
    })
    cfg = SimulationConfig.from_dict(cfg_data)
    assert not cfg.dt_user_provided
    assert _resolve_dt_ps(cfg) == pytest.approx(0.001)

    cfg_data = _base_config(tmp_path)
    cfg_data.update({
        "safe_mode": True,
        "dt": 0.004,
        "cmd_ns": 5.0,
        "equil_ns_per_cycle": 5.0,
        "prod_ns_per_cycle": 5.0,
        "heat_ns": 0.2,
        "density_ns": 0.5,
        "controller_enabled": False,
    })
    cfg = SimulationConfig.from_dict(cfg_data)
    assert cfg.dt_user_provided
    assert _resolve_dt_ps(cfg) == pytest.approx(0.004)
