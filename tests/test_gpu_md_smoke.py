import json
import math
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.config import SimulationConfig
from paddle.core.engine import EngineOptions, create_simulation, get_platform, minimize_and_initialize
from paddle.core.integrators import make_conventional


def _gpu_requested(pytestconfig: pytest.Config) -> bool:
    if os.environ.get("RUN_GPU_TESTS") == "1":
        return True
    markexpr = (pytestconfig.getoption("markexpr") or "").strip()
    if not markexpr:
        return False
    if "not gpu" in markexpr:
        return False
    return "gpu" in markexpr


def _require_cuda(openmm) -> None:
    names = {
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    }
    if "CUDA" not in names:
        pytest.fail(
            "CUDA platform not available; GPU tests require a CUDA-enabled OpenMM build."
        )


def _write_config(path: Path, parm_file: str, crd_file: str, outdir: Path) -> Path:
    payload = {
        "parmFile": parm_file,
        "crdFile": crd_file,
        "simType": "protein.implicit",
        "nbCutoff": 9.0,
        "temperature": 300.0,
        "ntcmd": 1000,
        "cmdRestartFreq": 200,
        "platform": "CUDA",
        "precision": "mixed",
        "deterministic_forces": False,
        "outdir": str(outdir),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_fallback_system(openmm):
    from openmm import CustomExternalForce, System, Vec3, unit
    from openmm.app import Topology, element

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    topology.addAtom("CA", element.carbon, residue)

    system = System()
    system.addParticle(12.0)
    force = CustomExternalForce("0.5 * k * (x^2 + y^2 + z^2)")
    force.addGlobalParameter("k", 100.0)
    force.addParticle(0, [])
    system.addForce(force)
    positions = [Vec3(0.0, 0.0, 0.0)] * unit.nanometer
    return topology, system, positions


@pytest.mark.gpu
def test_openmm_cuda_md_smoke(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    if not _gpu_requested(pytestconfig):
        pytest.skip("Set RUN_GPU_TESTS=1 or run pytest -m gpu to enable CUDA tests.")

    try:
        import openmm
        from openmm import XmlSerializer, unit
        from openmm.app import Simulation
    except Exception as exc:  # pragma: no cover - dependency import guard
        pytest.fail(f"OpenMM import failed for GPU test: {exc}")

    _require_cuda(openmm)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    alanine_root = ROOT / "benchmarks" / "alanine" / "implicit"
    alanine_parm = alanine_root / "complex.parm7"
    alanine_crd = alanine_root / "complex.rst7"

    if alanine_parm.exists() and alanine_crd.exists():
        parm_file = str(alanine_parm)
        crd_file = str(alanine_crd)
    else:
        parm_file = str(tmp_path / "placeholder.parm7")
        crd_file = str(tmp_path / "placeholder.rst7")
        Path(parm_file).touch()
        Path(crd_file).touch()

    config_path = _write_config(tmp_path / "gpu_config.json", parm_file, crd_file, outdir)
    cfg = SimulationConfig.from_file(config_path)

    integ = make_conventional(dt_ps=0.002, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    opts = EngineOptions(
        sim_type=cfg.simType,
        nb_cutoff_angstrom=cfg.nbCutoff,
        platform_name=cfg.platform,
        precision=cfg.precision,
        deterministic_forces=cfg.deterministic_forces,
        add_barostat=False,
    )

    if alanine_parm.exists() and alanine_crd.exists():
        sim = create_simulation(cfg.parmFile, cfg.crdFile, integ, opts)
        minimize_and_initialize(sim, cfg.temperature, set_velocities=True)
    else:
        topology, system, positions = _build_fallback_system(openmm)
        plat, props = get_platform(cfg.platform, cfg.precision, cfg.deterministic_forces)
        if plat.getName() != "CUDA":
            pytest.fail("CUDA platform not available; refusing CPU fallback in GPU test.")
        sim = Simulation(topology, system, integ, platform=plat, platformProperties=props)
        sim.context.setPositions(positions)
        sim.context.setVelocitiesToTemperature(cfg.temperature * unit.kelvin)

    platform_name = sim.context.getPlatform().getName()
    assert platform_name == "CUDA"

    sim.step(1000)

    state = sim.context.getState(getEnergy=True, getPositions=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert math.isfinite(potential_energy)

    state_path = outdir / "cmd_state.xml"
    state_path.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    assert state_path.exists()
