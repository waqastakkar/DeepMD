from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.config import SimulationConfig  # noqa: E402
from paddle.stages.equil_prod import _check_state_finite  # noqa: E402


class _FakeQuantity:
    def __init__(self, value):
        self._value = value

    def value_in_unit(self, unit):
        return self._value


class _FakeState:
    def __init__(self, positions, velocities, potential_energy, kinetic_energy=None, step=0):
        self._positions = positions
        self._velocities = velocities
        self._potential = potential_energy
        self._kinetic = kinetic_energy
        self._step = step

    def getPositions(self, asNumpy=False):
        return self._positions

    def getVelocities(self, asNumpy=False):
        return self._velocities

    def getPotentialEnergy(self):
        return self._potential

    def getKineticEnergy(self):
        return self._kinetic

    def getStepCount(self):
        return self._step


class _FakeContext:
    def __init__(self, state):
        self._state = state

    def getState(self, getPositions=False, getEnergy=False, getVelocities=False):
        return self._state


class _FakeSystem:
    def getNumParticles(self):
        return 1

    def getNumConstraints(self):
        return 0

    def usesPeriodicBoundaryConditions(self):
        return False


class _FakeSimulation:
    def __init__(self, state):
        self.context = _FakeContext(state)
        self.system = _FakeSystem()
        self.topology = object()


def test_check_state_finite_raises_on_nan_positions(tmp_path, monkeypatch):
    nan_positions = _FakeQuantity(np.array([[np.nan, 0.0, 0.0]]))
    velocities = _FakeQuantity(np.array([[0.0, 0.0, 0.0]]))
    potential = _FakeQuantity(10.0)
    kinetic = _FakeQuantity(5.0)
    state = _FakeState(nan_positions, velocities, potential, kinetic_energy=kinetic, step=12)
    sim = _FakeSimulation(state)
    cfg = SimulationConfig()

    monkeypatch.setattr(
        "paddle.stages.equil_prod.PDBFile.writeFile",
        lambda topology, positions, handle: handle.write("MODEL\nEND\n"),
    )

    with pytest.raises(RuntimeError):
        _check_state_finite(sim, cfg, "unit_test", tmp_path)

    crash_pdb = tmp_path / "crash_last_valid_unit_test.pdb"
    crash_cfg = tmp_path / "crash_config_unit_test.json"
    crash_diag = tmp_path / "crash_diagnostics_unit_test.json"
    assert crash_pdb.exists()
    assert crash_cfg.exists()
    assert crash_diag.exists()
