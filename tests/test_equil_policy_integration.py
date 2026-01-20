from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import importlib
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "io"))

openmm_stub = types.ModuleType("openmm")
openmm_stub.XmlSerializer = object()
openmm_stub.unit = types.SimpleNamespace(kilojoule_per_mole=1.0)
sys.modules.setdefault("openmm", openmm_stub)

openmm_app_stub = types.ModuleType("openmm.app")
openmm_app_stub.DCDReporter = object()
openmm_app_stub.StateDataReporter = object()
sys.modules.setdefault("openmm.app", openmm_app_stub)

validate_stub = types.ModuleType("validate")
metrics_stub = types.ModuleType("validate.metrics")
metrics_stub.gaussianity_report = lambda _data: {
    "skewness": 0.1,
    "excess_kurtosis": 0.2,
    "tail_risk": 0.01,
}
sys.modules.setdefault("validate", validate_stub)
sys.modules.setdefault("validate.metrics", metrics_stub)

paddle_stub = types.ModuleType("paddle")
sys.modules.setdefault("paddle", paddle_stub)

config_module = importlib.import_module("config")
sys.modules.setdefault("paddle.config", config_module)
setattr(paddle_stub, "config", config_module)

paddle_core_stub = types.ModuleType("paddle.core")
sys.modules.setdefault("paddle.core", paddle_core_stub)
setattr(paddle_stub, "core", paddle_core_stub)

paddle_io_stub = types.ModuleType("paddle.io")
sys.modules.setdefault("paddle.io", paddle_io_stub)
setattr(paddle_stub, "io", paddle_io_stub)

@dataclass
class EngineOptions:
    sim_type: str = "explicit"
    nb_cutoff_angstrom: float = 10.0
    platform_name: str = "CPU"
    precision: str = "single"
    deterministic_forces: bool = False
    add_barostat: bool = False
    barostat_pressure_atm: float = 1.0
    barostat_interval: int = 25

engine_stub = types.ModuleType("paddle.core.engine")
engine_stub.EngineOptions = EngineOptions
engine_stub.create_simulation = lambda *args, **kwargs: None
engine_stub.minimize_and_initialize = lambda *args, **kwargs: None
sys.modules.setdefault("paddle.core.engine", engine_stub)

integrators_stub = types.ModuleType("paddle.core.integrators")
integrators_stub.make_dual_equil = lambda *args, **kwargs: None
integrators_stub.make_dual_prod = lambda *args, **kwargs: None
integrators_stub.make_conventional = lambda *args, **kwargs: None
sys.modules.setdefault("paddle.core.integrators", integrators_stub)

report_module = importlib.import_module("report")
restart_module = importlib.import_module("restart")
sys.modules.setdefault("paddle.io.report", report_module)
sys.modules.setdefault("paddle.io.restart", restart_module)

from config import SimulationConfig  # noqa: E402
from core.params import BoostParams  # noqa: E402
from stages import equil_prod  # noqa: E402


class DummyIntegrator:
    def __init__(self, params: BoostParams) -> None:
        self.params = params
        self._globals = {
            "VminD": params.VminD,
            "VmaxD": params.VmaxD,
            "VminP": params.VminP,
            "VmaxP": params.VmaxP,
            "DihedralRefEnergy": params.VminD + 1.0,
            "TotalRefEnergy": params.VminP + 1.0,
            "DihedralBoostPotential": 0.0,
            "TotalBoostPotential": 0.0,
            "Dihedralk0": params.k0D,
            "Totalk0": params.k0P,
        }

    def getGlobalVariableByName(self, name: str) -> float:
        return float(self._globals[name])


class DummyReporter:
    def __init__(self, *args, **kwargs) -> None:
        pass


class DummySim:
    def __init__(self) -> None:
        self.reporters = []

    def step(self, _steps: int) -> None:
        return None


def test_equil_cycle_policy_integration(tmp_path, monkeypatch):
    cfg = SimulationConfig()
    cfg.ntebpercyc = 10
    cfg.ebRestartFreq = 1

    expected_params = BoostParams(
        VminD=1.0,
        VmaxD=2.0,
        VminP=3.0,
        VmaxP=4.0,
        k0D=0.62,
        k0P=0.44,
    )

    called = {}

    def fake_propose(cfg_arg, cycle_stats, last_restart, metrics, model_summary=None):
        called["policy"] = {
            "cycle_stats": dict(cycle_stats),
            "metrics": dict(metrics),
            "model_summary": model_summary,
        }
        return expected_params

    def fake_make_dual_equil(dt_ps, temperature_K, params):
        called["params"] = params
        return DummyIntegrator(params)

    monkeypatch.setattr(equil_prod, "propose_boost_params", fake_propose)
    monkeypatch.setattr(equil_prod, "make_dual_equil", fake_make_dual_equil)
    monkeypatch.setattr(equil_prod, "_attach_integrator", lambda sim, integrator: None)
    monkeypatch.setattr(
        equil_prod,
        "_estimate_bounds",
        lambda sim, steps, interval: (1.0, 2.0, 3.0, 4.0, [0.9, 1.1], [2.9, 3.1]),
    )
    monkeypatch.setattr(equil_prod, "DCDReporter", DummyReporter)
    monkeypatch.setattr(equil_prod, "StateDataReporter", DummyReporter)

    sim = DummySim()
    metrics: dict[str, object] = {}

    equil_prod._run_equil_cycle(
        cfg,
        1,
        sim,
        tmp_path,
        None,
        metrics,
        model_summary={"uncertainty": 0.1},
    )

    assert "policy" in called
    assert called["params"] is expected_params
    assert "skewness" in metrics
    assert "excess_kurtosis" in metrics
    assert "tail_risk" in metrics

    bias_path = tmp_path / "bias_plan_cycle_1.json"
    assert bias_path.exists()
    data = json.loads(bias_path.read_text(encoding="utf-8"))
    assert data["params"]["k0D"] == expected_params.k0D
    assert data["params"]["k0P"] == expected_params.k0P
