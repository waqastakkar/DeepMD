"""
stages/cmd.py — Conventional MD (CMD) stage for PADDLE
"""
from __future__ import annotations

from pathlib import Path
from openmm import XmlSerializer, unit
from openmm.app import DCDReporter, StateDataReporter

from paddle.config import SimulationConfig, is_explicit_simtype, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation, minimize_and_initialize
from paddle.core.integrators import make_conventional
from paddle.io.report import ensure_dir, write_run_manifest

def _options_from_cfg(cfg: SimulationConfig) -> EngineOptions:
    return EngineOptions(
        sim_type=cfg.simType,
        nb_cutoff_angstrom=cfg.nbCutoff,
        platform_name=cfg.platform,
        precision=cfg.precision,
        deterministic_forces=cfg.deterministic_forces,
        add_barostat=is_explicit_simtype(cfg.simType),
        barostat_pressure_atm=1.0,
        barostat_interval=25,
    )

def run_cmd(cfg: SimulationConfig) -> None:
    ensure_dir(Path(cfg.outdir))
    set_global_seed(cfg.seed)
    integ = make_conventional(dt_ps=0.002, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    opts = _options_from_cfg(cfg)
    sim = create_simulation(cfg.parmFile, cfg.crdFile, integ, opts)
    minimize_and_initialize(sim, cfg.temperature, set_velocities=True)

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

    sim.step(cfg.ntcmd)

    state = sim.context.getState(getPositions=True, getVelocities=True, getForces=False, getEnergy=False)
    with open(rst_path, "w", encoding="utf-8") as f:
        f.write(XmlSerializer.serialize(state))

    write_run_manifest(Path(cfg.outdir), {
        "stage": "cmd",
        "steps": cfg.ntcmd,
        "report_interval": cfg.cmdRestartFreq,
        "platform": opts.platform_name,
        "precision": opts.precision,
        "deterministic_forces": opts.deterministic_forces,
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
