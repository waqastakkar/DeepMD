"""Run a real OpenMM CUDA alanine dipeptide MD benchmark."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from paddle.bench.alanine_system import build_alanine_ace_ala_nme_system

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("MD test config must be a YAML mapping.")
    return payload


def _select_platform(platform_name: str, require_gpu: bool, device_index: int, precision: str):
    from openmm import Platform

    platform_names = {
        Platform.getPlatform(i).getName()
        for i in range(Platform.getNumPlatforms())
    }
    if platform_name not in platform_names:
        raise RuntimeError(
            f"Requested platform '{platform_name}' not available. Found: {sorted(platform_names)}"
        )

    platform = Platform.getPlatformByName(platform_name)
    if require_gpu and platform.getName() != "CUDA":
        raise RuntimeError("CUDA platform required but not available; refusing CPU fallback.")

    properties = {}
    if platform.getName() == "CUDA":
        properties["DeviceIndex"] = str(device_index)
        properties["Precision"] = precision
    return platform, properties


def _assert_finite_state(state, outdir: Path, label: str) -> None:
    from openmm import XmlSerializer, unit

    potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    energy_ok = math.isfinite(potential_energy)
    positions_ok = np.isfinite(positions).all()
    if energy_ok and positions_ok:
        return

    failed_path = outdir / "failed_state.xml"
    failed_path.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    raise RuntimeError(
        f"Detected non-finite {'energy' if not energy_ok else 'positions'} in {label} state."
    )


def run_alanine_mdtest(cfg_path: Path) -> Path:
    """Run the alanine dipeptide CUDA MD test and return the output directory."""
    cfg = _load_config(Path(cfg_path))
    outdir = Path(cfg.get("output_dir", "benchmarks/alanine_mdtest/out"))
    outdir.mkdir(parents=True, exist_ok=True)

    platform_name = str(cfg.get("platform", "CUDA"))
    require_gpu = bool(cfg.get("require_gpu", False))
    device_index = int(cfg.get("cuda_device_index", 0))
    precision = str(cfg.get("cuda_precision", "mixed"))

    temperature = float(cfg.get("temperature", 300.0))
    dt_fs = float(cfg.get("dt_fs", 2.0))
    friction = float(cfg.get("friction_per_ps", 1.0))
    steps_total = int(cfg.get("steps_total", 2500000))
    report_interval = int(cfg.get("report_interval", 10000))

    write_trajectory = bool(cfg.get("write_trajectory", False))
    trajectory_interval = int(cfg.get("trajectory_interval", report_interval))

    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import DCDReporter, Simulation, StateDataReporter

    platform, properties = _select_platform(platform_name, require_gpu, device_index, precision)

    topology, system, positions = build_alanine_ace_ala_nme_system()

    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        dt_fs * unit.femtosecond,
    )

    simulation = Simulation(
        topology,
        system,
        integrator,
        platform=platform,
        platformProperties=properties,
    )
    simulation.context.setPositions(positions)

    logger.info("Minimizing energy for alanine dipeptide test...")
    simulation.minimizeEnergy(maxIterations=5000)

    minimize_state = simulation.context.getState(getEnergy=True, getPositions=True)
    _assert_finite_state(minimize_state, outdir, "minimized")

    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    log_path = outdir / "md.log"
    simulation.reporters.append(
        StateDataReporter(
            str(log_path),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
            separator=",",
        )
    )

    if write_trajectory:
        dcd_path = outdir / "traj.dcd"
        simulation.reporters.append(DCDReporter(str(dcd_path), trajectory_interval))

    logger.info("Running %d steps of MD on %s...", steps_total, platform.getName())
    simulation.step(steps_total)

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    _assert_finite_state(final_state, outdir, "final")

    state_path = outdir / "state.xml"
    state_path.write_text(XmlSerializer.serialize(final_state), encoding="utf-8")

    metadata = {
        "platform": platform.getName(),
        "device_index": device_index,
        "precision": precision,
        "steps_total": steps_total,
        "report_interval": report_interval,
        "temperature": temperature,
        "dt_fs": dt_fs,
        "friction_per_ps": friction,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("MD test complete. Outputs written to %s", outdir)
    return outdir
