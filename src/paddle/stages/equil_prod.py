"""
stages/equil_prod.py — Equilibration (dual-boost) + Production
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
from openmm import XmlSerializer, unit
from openmm.app import DCDReporter, StateDataReporter
import openmm as mm
from paddle.config import SimulationConfig, is_explicit_simtype, set_global_seed
from paddle.core.engine import EngineOptions, create_simulation, log_simulation_start
from paddle.core.integrators import make_dual_equil, make_dual_prod, make_conventional
from paddle.io.report import ensure_dir, write_run_manifest, append_metrics, write_json
from paddle.io.restart import RestartRecord, read_restart, write_restart, record_to_boost_params, validate_against_state
from paddle.learn.data import load_latent_pca, project_pca
from paddle.stages.prep import run_density_equil, run_heating, run_minimization, transfer_state
from paddle.policy import (
    gaussian_confidence,
    freeze_bias_update,
    multi_objective_alpha,
    propose_boost_params,
    uncertainty_scale,
)
from paddle.validate.metrics import (
    aggregate_gaussianity,
    detect_change_point,
    exploration_proxy,
    exploration_score,
    gaussianity_report,
    reweighting_diagnostics,
)

_KB_KJ_MOL_K = 0.008314462618

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

def _resolve_reweight_beta(
    cfg: SimulationConfig,
    temperature_samples: list[float],
) -> tuple[float, str]:
    if temperature_samples:
        t_mean = float(np.mean(temperature_samples))
        if t_mean > 0.0:
            return 1.0 / (_KB_KJ_MOL_K * t_mean), "temperature"
    beta_cfg = getattr(cfg, "reweight_beta", None)
    if beta_cfg is not None:
        return float(beta_cfg), "config"
    return 1.0, "unit"

def _make_reweight_sampler(sim, integ, deltaV_samples: list[float], temperature_samples: list[float]):
    state = {"enabled": True, "available": True}
    def sample() -> None:
        if not state["enabled"]:
            return
        try:
            dboost = float(integ.getGlobalVariableByName("DihedralBoostPotential"))
            tboost = float(integ.getGlobalVariableByName("TotalBoostPotential"))
        except Exception:
            print("Warning: reweight diagnostics skipped (boost potentials unavailable).")
            state["enabled"] = False
            state["available"] = False
            return
        deltaV_samples.append(dboost + tboost)
        try:
            st = sim.context.getState(getEnergy=True, getVelocities=True)
            temp = _compute_temperature_k(st, sim)
        except Exception:
            temp = None
        if temp is not None:
            temperature_samples.append(float(temp))
    return sample, state


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

def _adaptive_stop_status(
    cfg: SimulationConfig,
    cyc: int,
    conf_hist: list[float],
    frozen_hist: list[bool],
    k0D_hist: list[float],
    k0P_hist: list[float],
    explore_hist: list[float],
) -> dict[str, object]:
    enabled = bool(getattr(cfg, "adaptive_stop_enabled", False))
    N = int(getattr(cfg, "adaptive_stop_cycles", 3))
    conf_threshold = float(getattr(cfg, "adaptive_stop_conf", 0.85))
    k0_tol = float(getattr(cfg, "adaptive_stop_k0_tol", 0.02))
    explore_tol = float(getattr(cfg, "adaptive_stop_explore_tol", 0.03))
    min_cycles = int(getattr(cfg, "adaptive_stop_min_cycles", 5))
    ready = enabled and (cyc + 1 >= min_cycles) and N > 0 and len(conf_hist) >= N
    triggered = False
    conf_tail: list[float] = []
    frozen_tail: list[bool] = []
    k0D_tail: list[float] = []
    k0P_tail: list[float] = []
    explore_tail: list[float] = []
    k0_delta_max = 0.0
    explore_delta_max = 0.0
    if ready:
        conf_tail = conf_hist[-N:]
        frozen_tail = frozen_hist[-N:]
        k0D_tail = k0D_hist[-N:]
        k0P_tail = k0P_hist[-N:]
        explore_tail = explore_hist[-N:]
        conf_ok = all(conf >= conf_threshold for conf in conf_tail)
        frozen_ok = all(not frozen for frozen in frozen_tail)
        k0_delta_max = max(
            [max(abs(k0D_tail[i] - k0D_tail[i - 1]), abs(k0P_tail[i] - k0P_tail[i - 1])) for i in range(1, N)]
            or [0.0]
        )
        explore_delta_max = max(
            [abs(explore_tail[i] - explore_tail[i - 1]) for i in range(1, N)]
            or [0.0]
        )
        k0_ok = k0_delta_max <= k0_tol
        explore_ok = explore_delta_max <= explore_tol
        triggered = bool(conf_ok and frozen_ok and k0_ok and explore_ok)
    return {
        "enabled": enabled,
        "triggered": triggered,
        "ready": ready,
        "N": N,
        "min_cycles": min_cycles,
        "conf_threshold": conf_threshold,
        "k0_tol": k0_tol,
        "explore_tol": explore_tol,
        "conf_tail": conf_tail,
        "frozen_tail": frozen_tail,
        "k0D_tail": k0D_tail,
        "k0P_tail": k0P_tail,
        "explore_tail": explore_tail,
        "k0_delta_max": k0_delta_max,
        "explore_delta_max": explore_delta_max,
    }


def _step_with_checks(
    sim,
    total_steps: int,
    block_size: int,
    label_prefix: str,
    outdir: Path,
    sample_fn=None,
) -> None:
    remaining = int(total_steps)
    block = max(1, int(block_size))
    block_index = 0
    while remaining > 0:
        step_now = min(block, remaining)
        sim.step(step_now)
        if sample_fn is not None:
            sample_fn()
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

def _resolve_kept_feature_indices(payload: dict[str, object]) -> Optional[list[int]]:
    kept = payload.get("kept_feature_indices")
    if kept is not None:
        return [int(i) for i in kept]
    original_dim = payload.get("original_dim")
    if original_dim is None:
        return None
    dropped = payload.get("dropped_constant_features", [])
    dropped_set = {int(i) for i in dropped}
    return [i for i in range(int(original_dim)) if i not in dropped_set]


def _validate_pca_projection_inputs(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    kept_feature_indices: Optional[Sequence[int]],
    original_dim: Optional[int],
) -> None:
    if X.ndim != 2:
        raise ValueError(f"PCA projection expects 2D X, got shape {X.shape}.")
    if original_dim is not None and X.shape[1] != original_dim:
        raise ValueError(
            "PCA projection input dimension mismatch before dropping: "
            f"X.shape={X.shape}, original_dim={original_dim}."
        )
    reduced_dim = len(kept_feature_indices) if kept_feature_indices is not None else X.shape[1]
    if reduced_dim != mean.shape[0] or reduced_dim != components.shape[1]:
        raise ValueError(
            "PCA projection dimension mismatch after dropping: "
            f"X.shape={X.shape}, mean.shape={mean.shape}, components.shape={components.shape}, "
            f"kept_feature_indices={list(kept_feature_indices) if kept_feature_indices is not None else None}."
        )


def _load_latent_pca_if_any(outdir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, dict[str, object]]]:
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

def _run_equil_cycle(
    cfg: SimulationConfig,
    cyc: int,
    sim,
    outdir: Path,
    last_restart: Optional[RestartRecord],
    metrics: dict[str, object],
    history_etot_mean: list[float],
    conf_hist: list[float],
    frozen_hist: list[bool],
    k0D_hist: list[float],
    k0P_hist: list[float],
    explore_hist: list[float],
    latent_pca: Optional[Tuple[np.ndarray, np.ndarray, dict[str, object]]] = None,
    model_summary: Optional[dict[str, object]] = None,
) -> tuple[RestartRecord, dict[str, object]]:
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

    exp = exploration_proxy(np.asarray(total_samples, dtype=float))
    metrics["explore_mean_abs_diff"] = float(exp["mean_abs_diff"])
    metrics["explore_std"] = float(exp["std"])

    latent_metrics: Optional[dict[str, float]] = None
    latent_explore_std: Optional[float] = None
    if latent_pca is not None:
        mean, components, payload = latent_pca
        kept_feature_indices = _resolve_kept_feature_indices(payload)
        original_dim = payload.get("original_dim")
        min_len = min(len(total_samples), len(dihedral_samples), len(temperature_samples))
        if min_len > 0:
            X_cycle = np.column_stack([
                np.asarray(total_samples[:min_len], dtype=float),
                np.asarray(dihedral_samples[:min_len], dtype=float),
                np.asarray(temperature_samples[:min_len], dtype=float),
            ])
            if bool(getattr(cfg, "safe_mode", False) or getattr(cfg, "validate_config", False)):
                _validate_pca_projection_inputs(
                    X_cycle,
                    mean,
                    components,
                    kept_feature_indices,
                    int(original_dim) if original_dim is not None else None,
                )
            Z = project_pca(X_cycle, mean, components, kept_feature_indices)
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
            if Z.shape[0] > 0 and Z.shape[1] > 0:
                latent_explore_std = float(np.std(Z[:, 0]))
            latent_metrics = {
                "latent_skewness": float(agg["skewness"]),
                "latent_excess_kurtosis": float(agg["excess_kurtosis"]),
                "latent_tail_risk": float(agg["tail_risk"]),
                "latent_gaussian_confidence": float(latent_conf),
            }
            metrics.update(latent_metrics)
    if latent_explore_std is not None:
        metrics["latent_explore_std"] = float(latent_explore_std)
    explore_std_good = float(getattr(cfg, "explore_std_good", 0.0))
    explore_std_high = float(getattr(cfg, "explore_std_high", 1.0))
    explore_score = exploration_score(metrics["explore_std"], explore_std_good, explore_std_high)
    if latent_explore_std is not None:
        latent_explore_score = exploration_score(latent_explore_std, explore_std_good, explore_std_high)
        explore_score = max(explore_score, latent_explore_score)
    metrics["explore_score"] = float(explore_score)
    history_etot_mean.append(etot_mean)
    cp = detect_change_point(
        np.array(history_etot_mean, dtype=float),
        window=int(getattr(cfg, "change_point_window", 5)),
        z_threshold=float(getattr(cfg, "change_point_z", 3.0)),
    )
    metrics["change_point"] = bool(cp["change_point"])
    metrics["change_point_z"] = float(cp["z_score"])
    metrics["gaussian_confidence"] = float(gaussian_confidence(cfg, metrics))
    metrics["controller_frozen"] = bool(freeze_bias_update(cfg, metrics))
    controller = {
        "gaussian_confidence": float(metrics["gaussian_confidence"]),
        "freeze_bias_update": bool(metrics["controller_frozen"]),
        "uncertainty_scale": float(uncertainty_scale(cfg, model_summary)),
        "controller_enabled": bool(getattr(cfg, "controller_enabled", True)),
        "change_point": bool(cp["change_point"]),
        "change_point_z": float(cp["z_score"]),
        "change_point_window": int(cp["window"]),
        "change_point_z_threshold": float(cp["z_threshold"]),
        "explore_mean_abs_diff": float(metrics["explore_mean_abs_diff"]),
        "explore_std": float(metrics["explore_std"]),
        "explore_score": float(metrics["explore_score"]),
    }
    if "conf_ewma" in metrics:
        controller["conf_ewma"] = float(metrics["conf_ewma"])
    if latent_explore_std is not None:
        controller["latent_explore_std"] = float(latent_explore_std)
    params = propose_boost_params(
        cfg,
        cycle_stats=cycle_stats,
        last_restart=last_restart,
        metrics=metrics,
        model_summary=model_summary,
    )
    controller["multi_objective_alpha"] = float(multi_objective_alpha(cfg, metrics, model_summary))
    if bool(getattr(cfg, "safe_mode", False)) and cyc == int(cfg.ncycebstart):
        params.k0D = 0.0
        params.k0P = 0.0
    conf_value = float(metrics.get("conf_ewma", metrics["gaussian_confidence"]))
    conf_hist.append(conf_value)
    k0D_hist.append(float(params.k0D))
    k0P_hist.append(float(params.k0P))
    explore_hist.append(float(metrics.get("explore_score", 0.0)))
    frozen_hist.append(bool(metrics.get("controller_frozen", freeze_bias_update(cfg, metrics))))
    adaptive_stop = _adaptive_stop_status(
        cfg,
        cyc,
        conf_hist,
        frozen_hist,
        k0D_hist,
        k0P_hist,
        explore_hist,
    )
    if adaptive_stop["enabled"]:
        controller["adaptive_stop_check"] = {
            "enabled": bool(adaptive_stop["enabled"]),
            "triggered": bool(adaptive_stop["triggered"]),
            "ready": bool(adaptive_stop["ready"]),
            "N": int(adaptive_stop["N"]),
            "min_cycles": int(adaptive_stop["min_cycles"]),
            "conf_threshold": float(adaptive_stop["conf_threshold"]),
            "k0_tol": float(adaptive_stop["k0_tol"]),
            "explore_tol": float(adaptive_stop["explore_tol"]),
            "k0_delta_max": float(adaptive_stop["k0_delta_max"]),
            "explore_delta_max": float(adaptive_stop["explore_delta_max"]),
        }
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

    reweight_enabled = bool(getattr(cfg, "reweight_diag_enabled", True))
    deltaV_samples: list[float] = []
    reweight_temperature_samples: list[float] = []
    sample_fn = None
    sample_state = None
    if reweight_enabled:
        sample_fn, sample_state = _make_reweight_sampler(
            sim,
            integ,
            deltaV_samples,
            reweight_temperature_samples,
        )

    _step_with_checks(
        sim,
        cfg.ntebpercyc,
        cfg.ebRestartFreq,
        f"equil_cycle{cyc:02d}",
        outdir,
        sample_fn=sample_fn,
    )

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
    if reweight_enabled:
        if sample_state is not None and not sample_state.get("available", True):
            pass
        elif len(deltaV_samples) < 2:
            print("Warning: reweight diagnostics skipped (no deltaV samples).")
        else:
            beta, beta_source = _resolve_reweight_beta(cfg, reweight_temperature_samples)
            reweight = reweighting_diagnostics(np.asarray(deltaV_samples, dtype=float), beta=beta)
            reweight["beta"] = float(beta)
            reweight["beta_source"] = beta_source
            metrics["reweight"] = reweight
            ess_min = float(getattr(cfg, "reweight_ess_min", 0.1))
            entropy_min = float(getattr(cfg, "reweight_entropy_min", 0.7))
            reweight_ok = bool(reweight["ess_frac"] >= ess_min and reweight["entropy_norm"] >= entropy_min)
            metrics["reweight_ok"] = reweight_ok
            bias_plan["metrics"]["reweight"] = reweight
            bias_plan["metrics"]["reweight_ok"] = reweight_ok
    write_json(bias_plan, outdir / f"bias_plan_cycle_{cyc}.json")
    write_restart(outdir / "gamd-restart.dat", rec)
    append_metrics({
        "phase": "equil","cycle": cyc,
        "VminD": VminD_f,"VmaxD": VmaxD_f,"VminP": VminP_f,"VmaxP": VmaxP_f,
        "k0D": k0D_f,"k0P": k0P_f
    }, outdir)
    return rec, adaptive_stop

def _run_prod_cycle(cfg: SimulationConfig, cyc: int, sim, outdir: Path, rec: RestartRecord) -> RestartRecord:
    prod_dir = outdir / "prod"
    ensure_dir(prod_dir)

    params = record_to_boost_params(rec)
    params.refED_factor = cfg.refED_factor
    params.refEP_factor = cfg.refEP_factor

    integ = make_dual_prod(dt_ps=_resolve_dt_ps(cfg), temperature_K=cfg.temperature, params=params)

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

    reweight_enabled = bool(getattr(cfg, "reweight_diag_enabled", True))
    deltaV_samples: list[float] = []
    reweight_temperature_samples: list[float] = []
    sample_fn = None
    sample_state = None
    if reweight_enabled:
        sample_fn, sample_state = _make_reweight_sampler(
            sim,
            integ,
            deltaV_samples,
            reweight_temperature_samples,
        )

    _step_with_checks(
        sim,
        cfg.ntprodpercyc,
        cfg.prodRestartFreq,
        f"prod_cycle{cyc:02d}",
        outdir,
        sample_fn=sample_fn,
    )

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
    prod_metrics = {"phase": "prod","cycle": cyc,"k0D": k0D_f,"k0P": k0P_f}
    if reweight_enabled:
        if sample_state is not None and not sample_state.get("available", True):
            pass
        elif len(deltaV_samples) < 2:
            print("Warning: reweight diagnostics skipped (no deltaV samples).")
        else:
            beta, beta_source = _resolve_reweight_beta(cfg, reweight_temperature_samples)
            reweight = reweighting_diagnostics(np.asarray(deltaV_samples, dtype=float), beta=beta)
            reweight["beta"] = float(beta)
            reweight["beta_source"] = beta_source
            ess_min = float(getattr(cfg, "reweight_ess_min", 0.1))
            entropy_min = float(getattr(cfg, "reweight_entropy_min", 0.7))
            reweight_ok = bool(reweight["ess_frac"] >= ess_min and reweight["entropy_norm"] >= entropy_min)
            prod_metrics["reweight"] = reweight
            prod_metrics["reweight_ok"] = reweight_ok
    append_metrics(prod_metrics, outdir)
    return rec2

def run_equil_and_prod(cfg: SimulationConfig) -> None:
    outdir = Path(cfg.outdir)
    ensure_dir(outdir)
    set_global_seed(cfg.seed)

    dt_ps = _resolve_dt_ps(cfg)
    logged = False
    cmd_checkpoint = outdir / "cmd.rst"
    loaded = False
    if cmd_checkpoint.exists():
        opts_npt = _options_from_cfg(cfg, add_barostat=True)
        integ0 = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
        sim = create_simulation(cfg.parmFile, cfg.crdFile, integ0, opts_npt)
        if not logged:
            log_simulation_start(
                stage="EQUIL+PROD",
                platform_name=sim.context.getPlatform().getName(),
                precision=opts_npt.cuda_precision if cfg.platform == "CUDA" else opts_npt.precision,
                deterministic_forces=opts_npt.deterministic_forces,
                dt_ps=dt_ps,
                ntcmd=cfg.ntcmd,
                ntprodpercyc=cfg.ntprodpercyc,
                explicit=is_explicit_simtype(cfg.simType),
                rigid_water=opts_npt.rigid_water,
                ewald_error_tolerance=opts_npt.ewald_error_tolerance,
            )
            logged = True
        loaded = _load_cmd_checkpoint_if_any(sim, outdir)
    if not loaded:
        opts_nvt = _options_from_cfg(cfg, add_barostat=False)
        integ_nvt = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
        sim_nvt = create_simulation(cfg.parmFile, cfg.crdFile, integ_nvt, opts_nvt)
        if not logged:
            log_simulation_start(
                stage="EQUIL+PROD",
                platform_name=sim_nvt.context.getPlatform().getName(),
                precision=opts_nvt.cuda_precision if cfg.platform == "CUDA" else opts_nvt.precision,
                deterministic_forces=opts_nvt.deterministic_forces,
                dt_ps=dt_ps,
                ntcmd=cfg.ntcmd,
                ntprodpercyc=cfg.ntprodpercyc,
                explicit=is_explicit_simtype(cfg.simType),
                rigid_water=opts_nvt.rigid_water,
                ewald_error_tolerance=opts_nvt.ewald_error_tolerance,
            )
            logged = True
        run_minimization(cfg, sim_nvt, outdir)
        run_heating(cfg, sim_nvt, outdir)
        if not cfg.do_heating:
            sim_nvt.context.setVelocitiesToTemperature(cfg.temperature * unit.kelvin)
        if is_explicit_simtype(cfg.simType):
            opts_npt = _options_from_cfg(cfg, add_barostat=True)
            integ0 = make_conventional(dt_ps=dt_ps, temperature_K=cfg.temperature, collision_rate_ps=1.0)
            sim = create_simulation(cfg.parmFile, cfg.crdFile, integ0, opts_npt)
            transfer_state(sim_nvt, sim)
        else:
            sim = sim_nvt
        run_density_equil(cfg, sim, outdir)
        _check_state_finite(sim, "prep", outdir)

    _assign_force_groups(sim)

    write_run_manifest(outdir, {
        "stage": "equil+prod",
        "equil_cycles": [int(cfg.ncycebstart), int(cfg.ncycebend) - 1],
        "prod_cycles": [int(cfg.ncycprodstart), int(cfg.ncycprodend) - 1],
        "ntebpercyc": int(cfg.ntebpercyc),
        "ntprodpercyc": int(cfg.ntprodpercyc),
        "temperature": float(cfg.temperature),
        "platform": sim.context.getPlatform().getName(),
        "precision": cfg.precision,
    })

    rec: Optional[RestartRecord] = None
    metrics: dict[str, object] = {}
    model_summary = _load_model_summary(outdir)
    latent_pca = _load_latent_pca_if_any(outdir)
    history_etot_mean: list[float] = []
    conf_hist: list[float] = []
    frozen_hist: list[bool] = []
    k0D_hist: list[float] = []
    k0P_hist: list[float] = []
    explore_hist: list[float] = []
    for cyc in range(int(cfg.ncycebstart), int(cfg.ncycebend)):
        rec, adaptive_stop = _run_equil_cycle(
            cfg,
            cyc,
            sim,
            outdir,
            rec,
            metrics,
            history_etot_mean,
            conf_hist,
            frozen_hist,
            k0D_hist,
            k0P_hist,
            explore_hist,
            latent_pca=latent_pca,
            model_summary=model_summary,
        )
        if adaptive_stop.get("enabled") and adaptive_stop.get("triggered"):
            stop_info = {
                "stop_cycle": int(cyc),
                "N": int(adaptive_stop["N"]),
                "thresholds": {
                    "conf": float(adaptive_stop["conf_threshold"]),
                    "k0_tol": float(adaptive_stop["k0_tol"]),
                    "explore_tol": float(adaptive_stop["explore_tol"]),
                    "min_cycles": int(adaptive_stop["min_cycles"]),
                },
                "history_tail": {
                    "conf": list(adaptive_stop["conf_tail"]),
                    "k0D": list(adaptive_stop["k0D_tail"]),
                    "k0P": list(adaptive_stop["k0P_tail"]),
                    "explore": list(adaptive_stop["explore_tail"]),
                },
            }
            write_json(stop_info, outdir / "adaptive_stop.json")
            print(f"[ADAPTIVE_STOP] Equilibration stopped early at cycle {cyc}")
            break

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
