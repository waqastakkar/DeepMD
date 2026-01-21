from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paddle.config import SimulationConfig  # noqa: E402
from paddle.policy import propose_boost_params  # noqa: E402
from paddle.io.restart import RestartRecord  # noqa: E402


def _make_restart(k0D: float, k0P: float) -> RestartRecord:
    return RestartRecord(
        steps=1,
        VminD_kJ=1.0,
        VmaxD_kJ=2.0,
        DihedralRef_kJ=1.5,
        DihedralBoost_kJ=0.0,
        k0D=k0D,
        VminP_kJ=3.0,
        VmaxP_kJ=4.0,
        TotalRef_kJ=3.5,
        TotalBoost_kJ=0.0,
        k0P=k0P,
    )


def test_policy_deterministic_mapping():
    cfg = SimulationConfig(k0_initial=0.5, k0_min=0.1, k0_max=0.9)
    cycle_stats = {"VminD": 0.0, "VmaxD": 10.0, "VminP": 5.0, "VmaxP": 15.0}
    metrics = {"skew": 0.05, "kurtosis": 0.05}
    model_summary = {"uncertainty": 0.05}

    params1 = propose_boost_params(cfg, cycle_stats, None, metrics, model_summary=model_summary)
    params2 = propose_boost_params(cfg, cycle_stats, None, metrics, model_summary=model_summary)

    assert params1.k0D == pytest.approx(params2.k0D)
    assert params1.k0P == pytest.approx(params2.k0P)
    assert params1.k0D == pytest.approx(0.55)
    assert params1.k0P == pytest.approx(0.55)


def test_policy_clamps_and_sanitizes_bounds():
    cfg = SimulationConfig(k0_initial=0.58, k0_min=0.1, k0_max=0.6)
    cycle_stats = {"VminD": 5.0, "VmaxD": 1.0, "VminP": 2.0, "VmaxP": 1.0}
    metrics = {"skew": 0.0, "kurtosis": 0.0}
    model_summary = {"uncertainty": 0.0}

    params = propose_boost_params(cfg, cycle_stats, None, metrics, model_summary=model_summary)

    assert cfg.k0_min <= params.k0D <= cfg.k0_max
    assert cfg.k0_min <= params.k0P <= cfg.k0_max
    assert params.VminD <= params.VmaxD
    assert params.VminP <= params.VmaxP
    assert params.k0D == pytest.approx(cfg.k0_max)


def test_policy_reduces_on_high_kurtosis():
    cfg = SimulationConfig(k0_initial=0.5, k0_min=0.1, k0_max=0.9)
    last_restart = _make_restart(0.4, 0.45)
    cycle_stats = {"VminD": 0.0, "VmaxD": 10.0, "VminP": 5.0, "VmaxP": 15.0}
    metrics = {"skew": 0.0, "kurtosis": 2.0}

    params = propose_boost_params(cfg, cycle_stats, last_restart, metrics)

    assert params.k0D < last_restart.k0D
    assert params.k0P < last_restart.k0P
