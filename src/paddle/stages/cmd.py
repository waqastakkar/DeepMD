"""
stages/cmd.py — Conventional MD (CMD) stage for PADDLE
"""
from __future__ import annotations

from pathlib import Path
from openmm import XmlSerializer, unit
from openmm.app import DCDReporter, StateDataReporter

from paddle.config import SimulationConfig, is_explicit_simtype, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation, log_simulation_start
from paddle.core.integrators import make_conventional
from paddle.io.report import ensure_dir, write_run_manifest
from paddle.stages.prep import run_density_equil, run_heating, run_minimization, transfer_state

def _options_from_cfg(cfg: SimulationConfig, *, add_barostat: bool) -> EngineOptions:
    return EngineOptions(
        sim_type=cfg.simType,
        nb_cutoff_angstrom=cfg.nbCutoff,
        platform_name=cfg.platform,
        precision=cfg.precision,
        cuda_precision=cfg.cuda_precision,
        cuda_device_index=cfg.cuda_device_index,
        deterministic_forces=cfg.deterministic_forces,
        add_barostat=add_barostat and is_explicit_simtype(cfg.simType),
        barostat_pressure_atm=cfg.pressure_atm,
        barostat_interval=cfg.barostat_interval,
        ewald_error_tolerance=cfg.ewaldErrorTolerance,
        use_dispersion_correction=cfg.useDispersionCorrection,
        rigid_water=cfg.rigidWater,
    )

def run_cmd(cfg: SimulationConfig) -> None:
    ensure_dir(Path(cfg.outdir))
    set_global_seed(cfg.seed)
    dt_ps = float(cfg.dt or 0.002)
    integ = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    opts_nvt = _options_from_cfg(cfg, add_barostat=False)
    sim = create_simulation(cfg.parmFile, cfg.crdFile, integ, opts_nvt)
    log_simulation_start(
        stage="CMD",
        platform_name=sim.context.getPlatform().getName(),
        precision=opts_nvt.cuda_precision if cfg.platform == "CUDA" else opts_nvt.precision,
        deterministic_forces=opts_nvt.deterministic_forces,
        dt_ps=dt_ps,
        ntcmd=cfg.ntcmd,
        ntprodpercyc=cfg.ntprodpercyc,
        explicit=is_explicit_simtype(cfg.simType),
        rigid_water=opts_nvt.rigid_water,
        ewald_error_tolerance=opts_nvt.ewald_error_tolerance,
    )
    run_minimization(cfg, sim, Path(cfg.outdir))
    run_heating(cfg, sim, Path(cfg.outdir))
    if not cfg.do_heating:
        sim.context.setVelocitiesToTemperature(cfg.temperature * unit.kelvin)
    if is_explicit_simtype(cfg.simType):
        opts_npt = _options_from_cfg(cfg, add_barostat=True)
        integ_npt = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
        sim_npt = create_simulation(cfg.parmFile, cfg.crdFile, integ_npt, opts_npt)
        transfer_state(sim, sim_npt)
        sim = sim_npt
    run_density_equil(cfg, sim, Path(cfg.outdir))

    dcd_path = Path(cfg.outdir) / "cmd.dcd"
    log_path = Path(cfg.outdir) / "cmd-state.log"
    md_log_path = Path(cfg.outdir) / "md.log"
    rst_path = Path(cfg.outdir) / "cmd.rst"

    sim.reporters.append(DCDReporter(str(dcd_path), cfg.cmdRestartFreq))
    sim.reporters.append(StateDataReporter(
        file=str(log_path), reportInterval=cfg.cmdRestartFreq,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=False, density=True, speed=True, separator="\t",
    ))
    sim.reporters.append(StateDataReporter(
        file=str(md_log_path), reportInterval=cfg.cmdRestartFreq,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=False, density=True, speed=True, separator="\t",
    ))

    total_ps = cfg.ntcmd * dt_ps
    total_ns = total_ps / 1000.0
    print(f"[CMD] ntcmd={cfg.ntcmd} dt_ps={dt_ps} total_ps={total_ps} total_ns={total_ns}")
    steps_advanced = 0
    chunk = max(1, int(cfg.cmdRestartFreq))
    remaining = int(cfg.ntcmd)
    while remaining > 0:
        n = min(chunk, remaining)
        sim.step(n)
        steps_advanced += n
        remaining -= n
    if cfg.safe_mode or cfg.validate_config:
        assert steps_advanced == cfg.ntcmd

    state = sim.context.getState(getPositions=True, getVelocities=True, getForces=False, getEnergy=False)
    with open(rst_path, "w", encoding="utf-8") as f:
        f.write(XmlSerializer.serialize(state))

    write_run_manifest(Path(cfg.outdir), {
        "stage": "cmd",
        "steps": cfg.ntcmd,
        "report_interval": cfg.cmdRestartFreq,
        "platform": sim.context.getPlatform().getName(),
        "precision": cfg.precision,
        "deterministic_forces": cfg.deterministic_forces,
        "simType": cfg.simType,
        "temperature": cfg.temperature,
    })

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="paddle CMD stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--ntcmd", type=int, default=None)
    ap.add_argument("--cmdRestartFreq", type=int, default=None)
    args = ap.parse_args()
    cfg = SimulationConfig.from_file(args.config)
    if args.outdir is not None: cfg.outdir = args.outdir
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.ntcmd is not None: cfg.ntcmd = args.ntcmd
    if args.cmdRestartFreq is not None: cfg.cmdRestartFreq = args.cmdRestartFreq
    cfg.validate()
    run_cmd(cfg)
    print(f"CMD stage complete. Outputs in: {cfg.outdir}")
