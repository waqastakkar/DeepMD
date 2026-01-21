"""
stages/equil_prod.py — Equilibration (dual-boost) + Production
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from openmm import XmlSerializer, unit
from openmm.app import DCDReporter, StateDataReporter
import openmm as mm
from paddle.config import SimulationConfig, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation
from paddle.core.integrators import make_dual_equil, make_dual_prod, make_conventional
from paddle.io.report import ensure_dir, write_run_manifest, append_metrics, write_json
from paddle.io.restart import RestartRecord, read_restart, write_restart, record_to_boost_params, validate_against_state
from paddle.learn.data import load_latent_pca, project_pca
from paddle.policy import (
    gaussian_confidence,
    freeze_bias_update,
    propose_boost_params,
    uncertainty_scale,
)
from paddle.validate.metrics import aggregate_gaussianity, detect_change_point, gaussianity_report

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

def _compute_temperature_k(state, sim) -> Optional[float]:
    try:
        kinetic = state.getKineticEnergy()
    except Exception:
        return None
    if kinetic is None:
        return None
    try:
        ndof = 3 * sim.system.getNumParticles()
        ndof -= sim.system.getNumConstraints()
        if sim.system.usesPeriodicBoundaryConditions():
            ndof -= 3
    except Exception:
        return None
    if ndof <= 0:
        return None
    try:
        temperature = (2 * kinetic) / (ndof * unit.MOLAR_GAS_CONSTANT_R)
        return float(temperature.value_in_unit(unit.kelvin))
    except Exception:
        return None


def _check_state_finite(sim, label: str, outdir: Path) -> None:
    ensure_dir(outdir)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True)
    positions_ok = False
    if positions is not None:
        positions_nm = positions.value_in_unit(unit.nanometer)
        positions_ok = bool(np.isfinite(positions_nm).all())
    potential = state.getPotentialEnergy()
    potential_kj = None
    potential_ok = False
    if potential is not None:
        potential_kj = float(potential.value_in_unit(unit.kilojoule_per_mole))
        potential_ok = bool(np.isfinite(potential_kj))
    if positions_ok and potential_ok:
        return
    failed_state = outdir / f"failed_state_{label}.xml"
    failed_state.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    diagnostics = {
        "step": int(getattr(state, "getStepCount", lambda: 0)()),
        "potential_energy_kj_mol": potential_kj,
        "temperature_k": _compute_temperature_k(state, sim),
    }
    write_json(diagnostics, outdir / f"failed_state_{label}.json")
    raise RuntimeError(
        f"Non-finite state detected ({label}). "
        f"Saved diagnostics to {failed_state}."
    )


def _resolve_dt_ps(cfg: SimulationConfig, default: float = 0.002) -> float:
    dt = getattr(cfg, "dt", None)
    if dt is None:
        return default
    return float(dt)


def _step_with_checks(sim, total_steps: int, block_size: int, label_prefix: str, outdir: Path) -> None:
    remaining = int(total_steps)
    block = max(1, int(block_size))
    block_index = 0
    while remaining > 0:
        step_now = min(block, remaining)
        sim.step(step_now)
        _check_state_finite(sim, f"{label_prefix}_block{block_index:04d}", outdir)
        remaining -= step_now
        block_index += 1


def _load_model_summary(outdir: Path) -> Optional[dict[str, object]]:
    summary_path = outdir / "model_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

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
    dihedral_samples: list[float] = []
    total_samples: list[float] = []
    temperature_samples: list[float] = []
    for _ in range(n):
        sim.step(interval)
        Ed = sim.context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        state_total = sim.context.getState(getEnergy=True, getVelocities=True)
        Ep = state_total.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        dihedral_samples.append(Ed)
        total_samples.append(Ep)
        temp = _compute_temperature_k(state_total, sim)
        if temp is not None:
            temperature_samples.append(float(temp))
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
    return vmin_d, vmax_d, vmin_p, vmax_p, dihedral_samples, total_samples, temperature_samples

def _load_latent_pca_if_any(outdir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    candidates = [
        outdir / "models" / "run1" / "latent_pca.json",
        outdir / "latent_pca.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return load_latent_pca(path)
            except Exception as exc:
                print(f"Warning: failed to load latent PCA from {path}: {exc}")
                return None
    return None

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
    history_etot_mean: list[float],
    latent_pca: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    model_summary: Optional[dict[str, object]] = None,
) -> RestartRecord:
    equil_dir = outdir / "equil"
    ensure_dir(equil_dir)

    VminD, VmaxD, VminP, VmaxP, dihedral_samples, total_samples, temperature_samples = _estimate_bounds(
        sim, steps=min(10000, cfg.ntebpercyc // 10), interval=max(10, cfg.ebRestartFreq)
    )
    etot_mean = float(np.mean(total_samples)) if total_samples else 0.0
    cycle_stats = {
        "VminD": VminD,
        "VmaxD": VmaxD,
        "VminP": VminP,
        "VmaxP": VmaxP,
        "Etot_mean": etot_mean,
    }
    dihedral_report = gaussianity_report(np.asarray(dihedral_samples, dtype=float))
    total_report = gaussianity_report(np.asarray(total_samples, dtype=float))
    gaussianity = {
        "skewness_dihedral": dihedral_report["skewness"],
        "excess_kurtosis_dihedral": dihedral_report["excess_kurtosis"],
        "tail_risk_dihedral": dihedral_report["tail_risk"],
        "skewness_total": total_report["skewness"],
        "excess_kurtosis_total": total_report["excess_kurtosis"],
        "tail_risk_total": total_report["tail_risk"],
    }
    gaussianity["skewness"] = 0.5 * (gaussianity["skewness_dihedral"] + gaussianity["skewness_total"])
    gaussianity["excess_kurtosis"] = 0.5 * (
        gaussianity["excess_kurtosis_dihedral"] + gaussianity["excess_kurtosis_total"]
    )
    gaussianity["tail_risk"] = 0.5 * (gaussianity["tail_risk_dihedral"] + gaussianity["tail_risk_total"])
    gaussianity["skew"] = gaussianity["skewness"]
    gaussianity["kurtosis"] = gaussianity["excess_kurtosis"]
    metrics.update(gaussianity)

    latent_metrics: Optional[dict[str, float]] = None
    if latent_pca is not None:
        mean, components = latent_pca
        min_len = min(len(total_samples), len(dihedral_samples), len(temperature_samples))
        if min_len > 0:
            X_cycle = np.column_stack([
                np.asarray(total_samples[:min_len], dtype=float),
                np.asarray(dihedral_samples[:min_len], dtype=float),
                np.asarray(temperature_samples[:min_len], dtype=float),
            ])
            Z = project_pca(X_cycle, mean, components)
            reports = [gaussianity_report(Z[:, i]) for i in range(Z.shape[1])]
            agg = aggregate_gaussianity(reports)
            latent_conf = gaussian_confidence(
                cfg,
                {
                    "skew": agg["skewness"],
                    "kurtosis": agg["excess_kurtosis"],
                    "tail_risk": agg["tail_risk"],
                },
            )
            latent_metrics = {
                "latent_skewness": float(agg["skewness"]),
                "latent_excess_kurtosis": float(agg["excess_kurtosis"]),
                "latent_tail_risk": float(agg["tail_risk"]),
                "latent_gaussian_confidence": float(latent_conf),
            }
            metrics.update(latent_metrics)
    history_etot_mean.append(etot_mean)
    cp = detect_change_point(
        np.array(history_etot_mean, dtype=float),
        window=int(getattr(cfg, "change_point_window", 5)),
        z_threshold=float(getattr(cfg, "change_point_z", 3.0)),
    )
    metrics["change_point"] = bool(cp["change_point"])
    metrics["change_point_z"] = float(cp["z_score"])
    controller = {
        "gaussian_confidence": float(gaussian_confidence(cfg, metrics)),
        "freeze_bias_update": bool(freeze_bias_update(cfg, metrics)),
        "uncertainty_scale": float(uncertainty_scale(cfg, model_summary)),
        "controller_enabled": bool(getattr(cfg, "controller_enabled", True)),
        "change_point": bool(cp["change_point"]),
        "change_point_z": float(cp["z_score"]),
        "change_point_window": int(cp["window"]),
        "change_point_z_threshold": float(cp["z_threshold"]),
    }
    params = propose_boost_params(
        cfg,
        cycle_stats=cycle_stats,
        last_restart=last_restart,
        metrics=metrics,
        model_summary=model_summary,
    )
    if bool(getattr(cfg, "safe_mode", False)) and cyc == int(cfg.ncycebstart):
        params.k0D = 0.0
        params.k0P = 0.0
    ref_ed = params.VminD + (params.VmaxD - params.VminD) / max(params.k0D, 1e-12)
    ref_ep = params.VminP + (params.VmaxP - params.VminP) / max(params.k0P, 1e-12)
    bias_plan = {
        "cycle": cyc,
        "params": {
            "VminD": params.VminD,
            "VmaxD": params.VmaxD,
            "VminP": params.VminP,
            "VmaxP": params.VmaxP,
            "k0D": params.k0D,
            "k0P": params.k0P,
            "refED_factor": params.refED_factor,
            "refEP_factor": params.refEP_factor,
            "refED": ref_ed,
            "refEP": ref_ep,
        },
        "metrics": {
            "cycle_stats": cycle_stats,
            **gaussianity,
        },
        "controller": controller,
    }
    if latent_metrics is not None:
        bias_plan["metrics"]["latent"] = {
            "skewness": latent_metrics["latent_skewness"],
            "excess_kurtosis": latent_metrics["latent_excess_kurtosis"],
            "tail_risk": latent_metrics["latent_tail_risk"],
            "gaussian_confidence": latent_metrics["latent_gaussian_confidence"],
        }
    if model_summary is not None:
        bias_plan["model_summary"] = model_summary
    write_json(bias_plan, outdir / f"bias_plan_cycle_{cyc}.json")

    integ = make_dual_equil(dt_ps=_resolve_dt_ps(cfg), temperature_K=cfg.temperature, params=params)

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

    _step_with_checks(sim, cfg.ntebpercyc, cfg.ebRestartFreq, f"equil_cycle{cyc:02d}", outdir)

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

    _step_with_checks(sim, cfg.ntprodpercyc, cfg.prodRestartFreq, f"prod_cycle{cyc:02d}", outdir)

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
        sim.minimizeEnergy()
        _check_state_finite(sim, "minimize", outdir)
        sim.context.setVelocitiesToTemperature(cfg.temperature * unit.kelvin)
        _check_state_finite(sim, "velocities", outdir)

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
    model_summary = _load_model_summary(outdir)
    latent_pca = _load_latent_pca_if_any(outdir)
    history_etot_mean: list[float] = []
    for cyc in range(int(cfg.ncycebstart), int(cfg.ncycebend)):
        rec = _run_equil_cycle(
            cfg,
            cyc,
            sim,
            outdir,
            rec,
            metrics,
            history_etot_mean,
            latent_pca=latent_pca,
            model_summary=model_summary,
        )

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
