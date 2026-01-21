"""
stages/prep.py — Minimization/heating/density preparation stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from openmm import unit

from paddle.config import SimulationConfig, ns_to_steps
from paddle.io.report import ensure_dir, timestamp, write_json


def transfer_state(src_sim, dst_sim) -> None:
    state = src_sim.context.getState(getPositions=True, getVelocities=True)
    dst_sim.context.setPositions(state.getPositions())
    velocities = state.getVelocities()
    if velocities is not None:
        dst_sim.context.setVelocities(velocities)
    try:
        a, b, c = src_sim.context.getPeriodicBoxVectors()
    except Exception:
        return
    try:
        dst_sim.context.setPeriodicBoxVectors(a, b, c)
    except Exception:
        pass


def _set_integrator_temperature(integrator, temperature_k: float) -> None:
    if hasattr(integrator, "setTemperature"):
        integrator.setTemperature(temperature_k * unit.kelvin)
        return
    thermal_energy = (
        unit.BOLTZMANN_CONSTANT_kB
        * unit.AVOGADRO_CONSTANT_NA
        * temperature_k
        * unit.kelvin
    )
    if hasattr(integrator, "setGlobalVariableByName"):
        integrator.setGlobalVariableByName("thermal_energy", thermal_energy)
        return
    try:
        index = integrator.getGlobalVariableIndex("thermal_energy")
    except Exception:
        index = 1
    integrator.setGlobalVariable(index, thermal_energy)


def _write_stage_summary(
    outdir: Path,
    name: str,
    *,
    steps: int,
    ns: float,
    temperature_start: Optional[float],
    temperature_end: Optional[float],
    final_density_g_ml: Optional[float],
    skipped: bool,
    started_at: str,
    ended_at: str,
) -> None:
    payload = {
        "stage": name,
        "steps": int(steps),
        "ns": float(ns),
        "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "final_density_g_ml": final_density_g_ml,
        "skipped": bool(skipped),
        "started_at": started_at,
        "ended_at": ended_at,
    }
    write_json(payload, outdir / f"prep_{name}.json")


def run_minimization(cfg: SimulationConfig, sim, outdir: Path) -> None:
    ensure_dir(outdir)
    started = timestamp()
    if not cfg.do_minimize:
        ended = timestamp()
        _write_stage_summary(
            outdir,
            "minimize",
            steps=0,
            ns=0.0,
            temperature_start=None,
            temperature_end=None,
            final_density_g_ml=None,
            skipped=True,
            started_at=started,
            ended_at=ended,
        )
        return

    tolerance = cfg.minimize_tolerance_kj_per_mol * unit.kilojoule_per_mole
    sim.minimizeEnergy(tolerance=tolerance, maxIterations=int(cfg.minimize_max_iter))
    ended = timestamp()
    _write_stage_summary(
        outdir,
        "minimize",
        steps=0,
        ns=0.0,
        temperature_start=None,
        temperature_end=None,
        final_density_g_ml=None,
        skipped=False,
        started_at=started,
        ended_at=ended,
    )


def run_heating(cfg: SimulationConfig, sim, outdir: Path) -> None:
    ensure_dir(outdir)
    started = timestamp()
    if not cfg.do_heating or cfg.heat_ns <= 0.0:
        ended = timestamp()
        _write_stage_summary(
            outdir,
            "heating",
            steps=0,
            ns=0.0,
            temperature_start=cfg.heat_t_start,
            temperature_end=cfg.heat_t_end,
            final_density_g_ml=None,
            skipped=True,
            started_at=started,
            ended_at=ended,
        )
        return

    dt_ps = float(cfg.dt or 0.002)
    total_steps = int(getattr(cfg, "ntheat", ns_to_steps(cfg.heat_ns, dt_ps)))
    chunk = max(1, int(cfg.heat_report_freq))
    t_start = float(cfg.heat_t_start)
    t_end = float(cfg.heat_t_end)

    sim.context.setVelocitiesToTemperature(t_start * unit.kelvin)
    steps_done = 0
    while steps_done < total_steps:
        n = min(chunk, total_steps - steps_done)
        progress = (steps_done + n) / float(total_steps)
        target_temp = t_start + (t_end - t_start) * progress
        _set_integrator_temperature(sim.integrator, target_temp)
        sim.step(n)
        steps_done += n

    ended = timestamp()
    _write_stage_summary(
        outdir,
        "heating",
        steps=total_steps,
        ns=cfg.heat_ns,
        temperature_start=t_start,
        temperature_end=t_end,
        final_density_g_ml=None,
        skipped=False,
        started_at=started,
        ended_at=ended,
    )


def run_density_equil(cfg: SimulationConfig, sim, outdir: Path) -> None:
    ensure_dir(outdir)
    started = timestamp()
    if not cfg.do_density_equil or cfg.density_ns <= 0.0:
        ended = timestamp()
        _write_stage_summary(
            outdir,
            "density",
            steps=0,
            ns=0.0,
            temperature_start=cfg.temperature,
            temperature_end=cfg.temperature,
            final_density_g_ml=None,
            skipped=True,
            started_at=started,
            ended_at=ended,
        )
        return

    dt_ps = float(cfg.dt or 0.002)
    total_steps = int(getattr(cfg, "ntdensity", ns_to_steps(cfg.density_ns, dt_ps)))
    chunk = max(1, int(cfg.density_report_freq))
    steps_done = 0
    while steps_done < total_steps:
        n = min(chunk, total_steps - steps_done)
        sim.step(n)
        steps_done += n

    density_value = None
    try:
        state = sim.context.getState(getEnergy=True)
        dens = state.getDensity()
        if dens is not None:
            density_value = float(dens.value_in_unit(unit.gram / unit.milliliter))
    except Exception:
        density_value = None

    ended = timestamp()
    _write_stage_summary(
        outdir,
        "density",
        steps=total_steps,
        ns=cfg.density_ns,
        temperature_start=cfg.temperature,
        temperature_end=cfg.temperature,
        final_density_g_ml=density_value,
        skipped=False,
        started_at=started,
        ended_at=ended,
    )
