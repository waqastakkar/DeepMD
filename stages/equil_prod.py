"""
stages/equil_prod.py — Equilibration (dual-boost) + Production
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from openmm import XmlSerializer, unit
from openmm.app import DCDReporter, StateDataReporter
import openmm as mm
from paddle.config import SimulationConfig, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation, minimize_and_initialize
from paddle.core.integrators import make_dual_equil, make_dual_prod, make_conventional
from paddle.io.report import ensure_dir, write_run_manifest, append_metrics
from paddle.io.restart import RestartRecord, read_restart, write_restart, record_to_boost_params, validate_against_state
from policy import propose_boost_params

# ---- NEW: helper to bind any newly created integrator to the existing Context
def _attach_integrator(sim, integrator) -> None:
    """
    Swap in a new integrator by rebuilding the Context and preserving state.
    This is the most robust way across OpenMM versions.
    """
    # Save current state
    st_posvel = sim.context.getState(getPositions=True, getVelocities=True)
    pos = st_posvel.getPositions()
    vel = st_posvel.getVelocities()
    try:
        a, b, c = sim.context.getPeriodicBoxVectors()
    except Exception:
        a = b = c = None

    # Build a fresh Context on the same Platform
    platform = sim.context.getPlatform()
    new_ctx = mm.Context(sim.system, integrator, platform)

    # Restore state
    if a is not None:
        try:
            new_ctx.setPeriodicBoxVectors(a, b, c)
        except Exception:
            pass
    new_ctx.setPositions(pos)
    if vel is not None:
        new_ctx.setVelocities(vel)

    # Swap into Simulation (keep both attributes in sync)
    sim.integrator = integrator
    sim.context = new_ctx

def _assign_force_groups(sim) -> None:
    for force in sim.system.getForces():
        if force.__class__.__name__ == "PeriodicTorsionForce":
            force.setForceGroup(2)

def _load_cmd_checkpoint_if_any(sim, outdir: Path) -> bool:
    rst = outdir / "cmd.rst"
    if not rst.exists():
        return False
    state = XmlSerializer.deserialize(rst.read_text(encoding="utf-8"))
    if state.getPositions() is not None:
        sim.context.setPositions(state.getPositions())
    if state.getVelocities() is not None:
        sim.context.setVelocities(state.getVelocities())
    try:
        a, b, c = state.getPeriodicBoxVectors()
        sim.context.setPeriodicBoxVectors(a, b, c)
    except Exception:
        pass
    return True

def _estimate_bounds(sim, steps: int = 10000, interval: int = 100):
    assert steps >= interval >= 1
    n = max(1, steps // interval)
    vmin_d = float("inf"); vmax_d = float("-inf")
    vmin_p = float("inf"); vmax_p = float("-inf")
    for _ in range(n):
        sim.step(interval)
        Ed = sim.context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        Ep = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if Ed < vmin_d: vmin_d = Ed
        if Ed > vmax_d: vmax_d = Ed
        if Ep < vmin_p: vmin_p = Ep
        if Ep > vmax_p: vmax_p = Ep
    pad_d = max(1.0, 0.02 * max(abs(vmax_d), 1.0))
    pad_p = max(1.0, 0.02 * max(abs(vmax_p), 1.0))
    if vmax_d - vmin_d < 1e-6:
        vmin_d -= pad_d; vmax_d += pad_d
    if vmax_p - vmin_p < 1e-6:
        vmin_p -= pad_p; vmax_p += pad_p
    return vmin_d, vmax_d, vmin_p, vmax_p

def _options_from_cfg(cfg: SimulationConfig) -> EngineOptions:
    return EngineOptions(
        sim_type=cfg.simType,
        nb_cutoff_angstrom=cfg.nbCutoff,
        platform_name=cfg.platform,
        precision=cfg.precision,
        deterministic_forces=cfg.deterministic_forces,
        add_barostat=(cfg.simType == "explicit"),
        barostat_pressure_atm=1.0,
        barostat_interval=25,
    )

def _run_equil_cycle(
    cfg: SimulationConfig,
    cyc: int,
    sim,
    outdir: Path,
    last_restart: Optional[RestartRecord],
    metrics: dict[str, object],
    model_summary: Optional[dict[str, object]] = None,
) -> RestartRecord:
    equil_dir = outdir / "equil"
    ensure_dir(equil_dir)

    VminD, VmaxD, VminP, VmaxP = _estimate_bounds(
        sim, steps=min(10000, cfg.ntebpercyc // 10), interval=max(10, cfg.ebRestartFreq)
    )
    cycle_stats = {"VminD": VminD, "VmaxD": VmaxD, "VminP": VminP, "VmaxP": VmaxP}
    params = propose_boost_params(
        cfg,
        cycle_stats=cycle_stats,
        last_restart=last_restart,
        metrics=metrics,
        model_summary=model_summary,
    )

    integ = make_dual_equil(dt_ps=0.002, temperature_K=cfg.temperature, params=params)

    # ---- NEW: bind the new integrator
    _attach_integrator(sim, integ)

    dcd = equil_dir / f"equil-cycle{cyc:02d}.dcd"
    log = equil_dir / f"equil-cycle{cyc:02d}.log"
    sim.reporters = []
    sim.reporters.append(DCDReporter(str(dcd), cfg.ebRestartFreq))
    sim.reporters.append(StateDataReporter(
        file=str(log), reportInterval=cfg.ebRestartFreq,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, density=True, speed=True, separator="\t",
    ))

    sim.step(cfg.ntebpercyc)

    VminD_f = float(integ.getGlobalVariableByName("VminD"))
    VmaxD_f = float(integ.getGlobalVariableByName("VmaxD"))
    VminP_f = float(integ.getGlobalVariableByName("VminP"))
    VmaxP_f = float(integ.getGlobalVariableByName("VmaxP"))
    Dref_f = float(integ.getGlobalVariableByName("DihedralRefEnergy"))
    Tref_f = float(integ.getGlobalVariableByName("TotalRefEnergy"))
    Dboost_f = float(integ.getGlobalVariableByName("DihedralBoostPotential"))
    Tboost_f = float(integ.getGlobalVariableByName("TotalBoostPotential"))
    k0D_f = float(integ.getGlobalVariableByName("Dihedralk0"))
    k0P_f = float(integ.getGlobalVariableByName("Totalk0"))

    rec = RestartRecord(
        steps=cfg.ntebpercyc,
        VminD_kJ=VminD_f, VmaxD_kJ=VmaxD_f, DihedralRef_kJ=Dref_f, DihedralBoost_kJ=Dboost_f, k0D=k0D_f,
        VminP_kJ=VminP_f, VmaxP_kJ=VmaxP_f, TotalRef_kJ=Tref_f, TotalBoost_kJ=Tboost_f, k0P=k0P_f,
    )
    write_restart(outdir / "gamd-restart.dat", rec)
    append_metrics({
        "phase": "equil","cycle": cyc,
        "VminD": VminD_f,"VmaxD": VmaxD_f,"VminP": VminP_f,"VmaxP": VmaxP_f,
        "k0D": k0D_f,"k0P": k0P_f
    }, outdir)
    return rec

def _run_prod_cycle(cfg: SimulationConfig, cyc: int, sim, outdir: Path, rec: RestartRecord) -> RestartRecord:
    prod_dir = outdir / "prod"
    ensure_dir(prod_dir)

    params = record_to_boost_params(rec)
    params.refED_factor = cfg.refED_factor
    params.refEP_factor = cfg.refEP_factor

    integ = make_dual_prod(dt_ps=0.002, temperature_K=cfg.temperature, params=params)

    # ---- NEW: bind the new integrator
    _attach_integrator(sim, integ)

    dcd = prod_dir / f"prod-cycle{cyc:02d}.dcd"
    log = prod_dir / f"prod-cycle{cyc:02d}.log"
    sim.reporters = []
    sim.reporters.append(DCDReporter(str(dcd), cfg.prodRestartFreq))
    sim.reporters.append(StateDataReporter(
        file=str(log), reportInterval=cfg.prodRestartFreq,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, density=True, speed=True, separator="\t",
    ))

    sim.step(cfg.ntprodpercyc)

    VminD_f = float(integ.getGlobalVariableByName("VminD"))
    VmaxD_f = float(integ.getGlobalVariableByName("VmaxD"))
    VminP_f = float(integ.getGlobalVariableByName("VminP"))
    VmaxP_f = float(integ.getGlobalVariableByName("VmaxP"))
    Dref_f = float(integ.getGlobalVariableByName("DihedralRefEnergy"))
    Tref_f = float(integ.getGlobalVariableByName("TotalRefEnergy"))
    Dboost_f = float(integ.getGlobalVariableByName("DihedralBoostPotential"))
    Tboost_f = float(integ.getGlobalVariableByName("TotalBoostPotential"))
    k0D_f = float(integ.getGlobalVariableByName("Dihedralk0"))
    k0P_f = float(integ.getGlobalVariableByName("Totalk0"))

    rec2 = RestartRecord(
        steps=cfg.ntprodpercyc,
        VminD_kJ=VminD_f, VmaxD_kJ=VmaxD_f, DihedralRef_kJ=Dref_f, DihedralBoost_kJ=Dboost_f, k0D=k0D_f,
        VminP_kJ=VminP_f, VmaxP_kJ=VmaxP_f, TotalRef_kJ=Tref_f, TotalBoost_kJ=Tboost_f, k0P=k0P_f,
    )
    write_restart(outdir / "gamd-restart.dat", rec2)
    append_metrics({"phase": "prod","cycle": cyc,"k0D": k0D_f,"k0P": k0P_f}, outdir)
    return rec2

def run_equil_and_prod(cfg: SimulationConfig) -> None:
    outdir = Path(cfg.outdir)
    ensure_dir(outdir)
    set_global_seed(cfg.seed)

    opts = _options_from_cfg(cfg)
    integ0 = make_conventional(dt_ps=0.002, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    sim = create_simulation(cfg.parmFile, cfg.crdFile, integ0, opts)
    _assign_force_groups(sim)

    loaded = _load_cmd_checkpoint_if_any(sim, outdir)
    if not loaded:
        minimize_and_initialize(sim, cfg.temperature, set_velocities=True)

    write_run_manifest(outdir, {
        "stage": "equil+prod",
        "equil_cycles": [int(cfg.ncycebstart), int(cfg.ncycebend) - 1],
        "prod_cycles": [int(cfg.ncycprodstart), int(cfg.ncycprodend) - 1],
        "ntebpercyc": int(cfg.ntebpercyc),
        "ntprodpercyc": int(cfg.ntprodpercyc),
        "temperature": float(cfg.temperature),
        "platform": opts.platform_name,
        "precision": opts.precision,
    })

    rec: Optional[RestartRecord] = None
    metrics: dict[str, object] = {}
    model_summary: Optional[dict[str, object]] = None
    for cyc in range(int(cfg.ncycebstart), int(cfg.ncycebend)):
        rec = _run_equil_cycle(cfg, cyc, sim, outdir, rec, metrics, model_summary=model_summary)

    if rec is None:
        restart_path = outdir / "gamd-restart.dat"
        rec = read_restart(restart_path)
        validate_against_state(rec)

    for cyc in range(int(cfg.ncycprodstart), int(cfg.ncycprodend)):
        rec = _run_prod_cycle(cfg, cyc, sim, outdir, rec)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="paddle equilibration + production stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--ncycebstart", type=int, default=None)
    ap.add_argument("--ncycebend", type=int, default=None)
    ap.add_argument("--ntebpercyc", type=int, default=None)
    ap.add_argument("--ebRestartFreq", type=int, default=None)
    ap.add_argument("--ncycprodstart", type=int, default=None)
    ap.add_argument("--ncycprodend", type=int, default=None)
    ap.add_argument("--ntprodpercyc", type=int, default=None)
    ap.add_argument("--prodRestartFreq", type=int, default=None)
    args = ap.parse_args()
    cfg = SimulationConfig.from_file(args.config)
    if args.outdir is not None: cfg.outdir = args.outdir
    if args.ncycebstart is not None: cfg.ncycebstart = args.ncycebstart
    if args.ncycebend is not None: cfg.ncycebend = args.ncycebend
    if args.ntebpercyc is not None: cfg.ntebpercyc = args.ntebpercyc
    if args.ebRestartFreq is not None: cfg.ebRestartFreq = args.ebRestartFreq
    if args.ncycprodstart is not None: cfg.ncycprodstart = args.ncycprodstart
    if args.ncycprodend is not None: cfg.ncycprodend = args.ncycprodend
    if args.ntprodpercyc is not None: cfg.ntprodpercyc = args.ntprodpercyc
    if args.prodRestartFreq is not None: cfg.prodRestartFreq = args.prodRestartFreq
    cfg.validate()
    run_equil_and_prod(cfg)
    print(f"Equilibration + Production complete. Outputs in: {cfg.outdir}")
