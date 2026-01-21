"""
stages/equil_prep.py — Equilibration preparation (collect energy series)
"""
from __future__ import annotations

from pathlib import Path
from openmm import XmlSerializer, unit

from paddle.config import SimulationConfig, is_explicit_simtype, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation, log_simulation_start, minimize_and_initialize
from paddle.core.integrators import make_conventional
from paddle.io.report import CSVLogger, ensure_dir, write_run_manifest, append_metrics

def _options_from_cfg(cfg: SimulationConfig) -> EngineOptions:
    return EngineOptions(
        sim_type=cfg.simType,
        nb_cutoff_angstrom=cfg.nbCutoff,
        platform_name=cfg.platform,
        precision=cfg.precision,
        cuda_precision=cfg.cuda_precision,
        cuda_device_index=cfg.cuda_device_index,
        deterministic_forces=cfg.deterministic_forces,
        add_barostat=is_explicit_simtype(cfg.simType),
        barostat_pressure_atm=1.0,
        barostat_interval=25,
        ewald_error_tolerance=cfg.ewaldErrorTolerance,
        use_dispersion_correction=cfg.useDispersionCorrection,
        rigid_water=cfg.rigidWater,
    )

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

def run_equil_prep(cfg: SimulationConfig) -> None:
    outdir = Path(cfg.outdir)
    prepdir = outdir / "prep"
    ensure_dir(prepdir)

    set_global_seed(cfg.seed)

    dt_ps = float(cfg.dt or 0.002)
    integ = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    opts = _options_from_cfg(cfg)
    sim = create_simulation(cfg.parmFile, cfg.crdFile, integ, opts)
    log_simulation_start(
        stage="EQUIL_PREP",
        platform_name=sim.context.getPlatform().getName(),
        precision=opts.cuda_precision if cfg.platform == "CUDA" else opts.precision,
        deterministic_forces=opts.deterministic_forces,
        dt_ps=dt_ps,
        ntcmd=cfg.ntcmd,
        ntprodpercyc=cfg.ntprodpercyc,
        explicit=is_explicit_simtype(cfg.simType),
        rigid_water=opts.rigid_water,
        ewald_error_tolerance=opts.ewald_error_tolerance,
    )

    _assign_force_groups(sim)
    loaded = _load_cmd_checkpoint_if_any(sim, outdir)
    if not loaded:
        minimize_and_initialize(sim, cfg.temperature, set_velocities=True)

    start_cyc = int(cfg.ncycebprepstart)
    end_cyc = int(cfg.ncycebprepend)

    write_run_manifest(prepdir, {
        "stage": "equil_prep",
        "cycles": [start_cyc, end_cyc - 1],
        "ntebpreppercyc": cfg.ntebpreppercyc,
        "report_interval": cfg.ebprepRestartFreq,
        "temperature": cfg.temperature,
        "platform": opts.platform_name,
        "precision": opts.precision,
    })

    for cyc in range(start_cyc, end_cyc):
        csv_path = prepdir / f"equilprep-cycle{cyc:02d}.csv"
        logger = CSVLogger(csv_path, ["step", "Etot_kJ", "Edih_kJ", "T_K"], compress=True)

        steps = cfg.ntebpreppercyc
        interval = max(1, int(cfg.ebprepRestartFreq))
        n_reports = steps // interval

        for i in range(n_reports):
            sim.step(interval)
            Etot = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            Edih = sim.context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            state = sim.context.getState(getEnergy=True)
            state = sim.context.getState(getEnergy=True)
            K_kJ = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            N = sim.system.getNumParticles()
            num_constraints = sim.system.getNumConstraints()  # counts constrained DOFs (e.g., HBonds)
            dof = max(1, 3 * N - num_constraints - 3)        # remove 3 for translation
            kB_kJ_per_mol_K = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.kilojoule_per_mole / unit.kelvin
            )
            T = 2.0 * K_kJ / (dof * kB_kJ_per_mol_K)
            logger.writerow({
                "step": (i + 1) * interval,
                "Etot_kJ": Etot,
                "Edih_kJ": Edih,
                "T_K": T,
            })

        append_metrics({"phase": "equil_prep","cycle": cyc,"samples": int(n_reports)}, prepdir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="paddle equilibration preparation stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--ncycebprepstart", type=int, default=None)
    ap.add_argument("--ncycebprepend", type=int, default=None)
    ap.add_argument("--ntebpreppercyc", type=int, default=None)
    ap.add_argument("--ebprepRestartFreq", type=int, default=None)
    args = ap.parse_args()
    cfg = SimulationConfig.from_file(args.config)
    if args.outdir is not None: cfg.outdir = args.outdir
    if args.ncycebprepstart is not None: cfg.ncycebprepstart = args.ncycebprepstart
    if args.ncycebprepend is not None: cfg.ncycebprepend = args.ncycebprepend
    if args.ntebpreppercyc is not None: cfg.ntebpreppercyc = args.ntebpreppercyc
    if args.ebprepRestartFreq is not None: cfg.ebprepRestartFreq = args.ebprepRestartFreq
    cfg.validate()
    run_equil_prep(cfg)
    print(f"Equil-prep complete. Outputs in: {cfg.outdir}/prep")
