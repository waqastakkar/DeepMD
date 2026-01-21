from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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


def _install_stubs(monkeypatch) -> None:
    openmm_stub = types.ModuleType("openmm")
    openmm_stub.XmlSerializer = object()
    openmm_stub.unit = types.SimpleNamespace(kilojoule_per_mole=1.0)
    monkeypatch.setitem(sys.modules, "openmm", openmm_stub)

    openmm_app_stub = types.ModuleType("openmm.app")
    openmm_app_stub.DCDReporter = object()
    openmm_app_stub.StateDataReporter = object()
    monkeypatch.setitem(sys.modules, "openmm.app", openmm_app_stub)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.asarray = lambda data, dtype=None: data
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    validate_stub = types.ModuleType("paddle.validate")
    validate_stub.__path__ = []
    metrics_stub = types.ModuleType("paddle.validate.metrics")
    metrics_stub.gaussianity_report = lambda _data: {
        "skewness": 0.1,
        "excess_kurtosis": 0.2,
        "tail_risk": 0.01,
    }
    monkeypatch.setitem(sys.modules, "paddle.validate", validate_stub)
    monkeypatch.setitem(sys.modules, "paddle.validate.metrics", metrics_stub)

    engine_stub = types.ModuleType("paddle.core.engine")
    engine_stub.EngineOptions = EngineOptions
    engine_stub.create_simulation = lambda *args, **kwargs: None
    engine_stub.minimize_and_initialize = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "paddle.core.engine", engine_stub)

    integrators_stub = types.ModuleType("paddle.core.integrators")
    integrators_stub.make_dual_equil = lambda *args, **kwargs: None
    integrators_stub.make_dual_prod = lambda *args, **kwargs: None
    integrators_stub.make_conventional = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "paddle.core.integrators", integrators_stub)


def _load_modules(monkeypatch):
    _install_stubs(monkeypatch)
    from paddle.config import SimulationConfig
    from paddle.core.params import BoostParams
    from paddle.io.restart import RestartRecord
    from paddle.stages import equil_prod

    return SimulationConfig, BoostParams, RestartRecord, equil_prod


def test_equil_cycle_policy_integration(tmp_path, monkeypatch):
    SimulationConfig, BoostParams, _, equil_prod = _load_modules(monkeypatch)

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


def test_run_equil_and_prod_loads_model_summary(tmp_path, monkeypatch):
    SimulationConfig, _, RestartRecord, equil_prod = _load_modules(monkeypatch)

    cfg = SimulationConfig()
    cfg.outdir = str(tmp_path)
    cfg.ncycebstart = 0
    cfg.ncycebend = 1
    cfg.ncycprodstart = 0
    cfg.ncycprodend = 0

    summary = {"uncertainty": 0.12, "calibration_score": 0.9}
    (tmp_path / "model_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    captured = {}

    def fake_run_equil_cycle(cfg_arg, cyc, sim, outdir, last_restart, metrics, model_summary=None):
        captured["model_summary"] = model_summary
        return RestartRecord(
            steps=1,
            VminD_kJ=1.0,
            VmaxD_kJ=2.0,
            DihedralRef_kJ=1.5,
            DihedralBoost_kJ=0.1,
            k0D=0.1,
            VminP_kJ=3.0,
            VmaxP_kJ=4.0,
            TotalRef_kJ=3.5,
            TotalBoost_kJ=0.2,
            k0P=0.2,
        )

    def fake_create_simulation(*args, **kwargs):
        return types.SimpleNamespace(
            reporters=[],
            step=lambda _steps: None,
            context=types.SimpleNamespace(
                getState=lambda **_kwargs: types.SimpleNamespace(
                    getPositions=lambda: None,
                    getVelocities=lambda: None,
                    getPeriodicBoxVectors=lambda: (None, None, None),
                ),
                setPositions=lambda _pos: None,
                setVelocities=lambda _vel: None,
                setPeriodicBoxVectors=lambda *_args: None,
                getPlatform=lambda: None,
            ),
            system=types.SimpleNamespace(getForces=lambda: []),
        )

    monkeypatch.setattr(equil_prod, "_load_cmd_checkpoint_if_any", lambda *args, **kwargs: True)
    monkeypatch.setattr(equil_prod, "_run_equil_cycle", fake_run_equil_cycle)
    monkeypatch.setattr(equil_prod, "_run_prod_cycle", lambda *args, **kwargs: None)
    monkeypatch.setattr(equil_prod, "create_simulation", fake_create_simulation)
    monkeypatch.setattr(equil_prod, "minimize_and_initialize", lambda *args, **kwargs: None)

    equil_prod.run_equil_and_prod(cfg)

    assert captured["model_summary"] == summary
