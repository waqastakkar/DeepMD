"""
policy.py — Rule-based closed-loop bias planning
"""
from __future__ import annotations

from typing import Mapping, Optional

from paddle.core.params import BoostParams

_KURTOSIS_HIGH = 1.0
_KURTOSIS_GOOD = 0.2
_SKEW_GOOD = 0.2
_UNCERTAINTY_LOW = 0.2
_K0_DOWN_FACTOR = 0.8
_K0_UP_STEP = 0.05
_MIN_SPAN = 1e-6
_KURTOSIS_FREEZE = 2.0
_SKEW_FREEZE = 0.6
_TAIL_RISK_GOOD = 0.01
_TAIL_RISK_FREEZE = 0.05
_EPS = 1e-12


def _get_value(
    source: object,
    key: str,
    default: Optional[float] = None,
    required: bool = False,
) -> Optional[float]:
    if source is None:
        if required and default is None:
            raise ValueError(f"Missing required config key: {key}. Please set it in YAML.")
        return float(default) if default is not None else None
    if isinstance(source, Mapping):
        if key not in source or source[key] is None:
            if required and default is None:
                raise ValueError(f"Missing required config key: {key}. Please set it in YAML.")
            return float(default) if default is not None else None
        return float(source[key])
    if not hasattr(source, key) or getattr(source, key) is None:
        if required and default is None:
            raise ValueError(f"Missing required config key: {key}. Please set it in YAML.")
        return float(default) if default is not None else None
    return float(getattr(source, key))


def _missing_keys(source: object, keys: list[str]) -> list[str]:
    missing: list[str] = []
    for key in keys:
        if source is None:
            missing.append(key)
            continue
        if isinstance(source, Mapping):
            if key not in source or source[key] is None:
                missing.append(key)
            continue
        if not hasattr(source, key) or getattr(source, key) is None:
            missing.append(key)
    return missing


def _infer_boost_mode(cfg) -> Optional[str]:
    for key in ("gamd_mode", "gamd_boost_mode", "boost_mode"):
        if hasattr(cfg, key):
            value = str(getattr(cfg, key)).lower()
            if "dihedral" in value and "dual" not in value and "total" not in value:
                return "dihedral"
            if "dual" in value or "total" in value:
                return "dual"
    for key in ("dihedral_only", "dihedral_boost_only", "gamd_dihedral_only"):
        if hasattr(cfg, key):
            return "dihedral" if bool(getattr(cfg, key)) else "dual"
    return None


def validate_config(cfg) -> None:
    if not bool(getattr(cfg, "controller_enabled", True)):
        return
    required_keys = [
        "k0_initial",
        "k0_min",
        "k0_max",
        "deltaV_damp_factor",
        "gaussian_skew_good",
        "gaussian_excess_kurtosis_good",
        "gaussian_tail_risk_good",
    ]
    missing = _missing_keys(cfg, required_keys)
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            "Missing required config keys for GaMD controller: "
            f"{missing_list}. Please set them in YAML."
        )


def _metric(metrics: Mapping[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _sanitize_bounds(vmin: float, vmax: float) -> tuple[float, float]:
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmax - vmin < _MIN_SPAN:
        pad = max(1.0, 0.02 * max(abs(vmax), 1.0))
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _linear_confidence(value: Optional[float], good: float, freeze: float) -> float:
    if value is None:
        return 0.0
    span = freeze - good
    if span <= 0:
        return 0.0 if abs(value) >= freeze else 1.0
    if abs(value) <= good:
        return 1.0
    if abs(value) >= freeze:
        return 0.0
    return (freeze - abs(value)) / span

def _gaussian_confidence_from_values(
    cfg,
    skew: Optional[float],
    kurtosis: Optional[float],
    tail: Optional[float],
) -> float:
    skew_good = float(getattr(cfg, "gaussian_skew_good", _SKEW_GOOD))
    skew_freeze = float(getattr(cfg, "gaussian_skew_freeze", _SKEW_FREEZE))
    kurtosis_good = float(getattr(cfg, "gaussian_excess_kurtosis_good", _KURTOSIS_GOOD))
    kurtosis_freeze = float(getattr(cfg, "gaussian_excess_kurtosis_freeze", _KURTOSIS_FREEZE))
    tail_good = float(getattr(cfg, "gaussian_tail_risk_good", _TAIL_RISK_GOOD))
    tail_freeze = float(getattr(cfg, "gaussian_tail_risk_freeze", _TAIL_RISK_FREEZE))

    conf_skew = _linear_confidence(skew, skew_good, skew_freeze)
    conf_kurt = _linear_confidence(kurtosis, kurtosis_good, kurtosis_freeze)
    conf_tail = _linear_confidence(tail, tail_good, tail_freeze)
    return min(conf_skew, conf_kurt, conf_tail)


def _uncertainty_is_low(model_summary: Optional[Mapping[str, object]]) -> bool:
    if model_summary is None:
        return True
    value = _metric(model_summary, "uncertainty", "sigma", "std", "variance")
    if value is None:
        return False
    return abs(value) <= _UNCERTAINTY_LOW


def gaussian_confidence(cfg, metrics: Mapping[str, object]) -> float:
    """Return a conservative Gaussianity confidence in [0, 1]."""
    skew = _metric(metrics, "skew", "skewness", "skew_proxy", "skewness_proxy")
    kurtosis = _metric(metrics, "kurtosis", "excess_kurtosis", "kurtosis_proxy", "excess_kurtosis_proxy")
    tail = _metric(metrics, "tail_risk", "tail_risk_proxy")
    energy_conf = _gaussian_confidence_from_values(cfg, skew, kurtosis, tail)

    latent_skew = _metric(metrics, "latent_skewness", "latent_skew")
    latent_kurtosis = _metric(metrics, "latent_excess_kurtosis", "latent_kurtosis")
    latent_tail = _metric(metrics, "latent_tail_risk")
    if latent_skew is not None and latent_kurtosis is not None and latent_tail is not None:
        latent_conf = _gaussian_confidence_from_values(cfg, latent_skew, latent_kurtosis, latent_tail)
        return min(energy_conf, latent_conf)
    return energy_conf


def freeze_bias_update(cfg, metrics: Mapping[str, object]) -> bool:
    """Decide whether to freeze bias updates based on Gaussianity."""
    if bool(metrics.get("change_point", False)):
        return True
    skew_freeze = float(getattr(cfg, "gaussian_skew_freeze", _SKEW_FREEZE))
    kurtosis_freeze = float(getattr(cfg, "gaussian_excess_kurtosis_freeze", _KURTOSIS_FREEZE))
    tail_freeze = float(getattr(cfg, "gaussian_tail_risk_freeze", _TAIL_RISK_FREEZE))

    skew = _metric(metrics, "skew", "skewness", "skew_proxy", "skewness_proxy")
    kurtosis = _metric(metrics, "kurtosis", "excess_kurtosis", "kurtosis_proxy", "excess_kurtosis_proxy")
    tail = _metric(metrics, "tail_risk", "tail_risk_proxy")
    delta_tail = _metric(metrics, "deltaV_tail_risk")
    if skew is None or kurtosis is None or tail is None:
        energy_freeze = True
    else:
        energy_freeze = (
            abs(skew) >= skew_freeze
            or abs(kurtosis) >= kurtosis_freeze
            or abs(tail) >= tail_freeze
        )
    if delta_tail is not None:
        energy_freeze = energy_freeze or abs(delta_tail) >= tail_freeze

    latent_skew = _metric(metrics, "latent_skewness", "latent_skew")
    latent_kurtosis = _metric(metrics, "latent_excess_kurtosis", "latent_kurtosis")
    latent_tail = _metric(metrics, "latent_tail_risk")
    latent_freeze = False
    if latent_skew is not None and latent_kurtosis is not None and latent_tail is not None:
        latent_freeze = (
            abs(latent_skew) >= skew_freeze
            or abs(latent_kurtosis) >= kurtosis_freeze
            or abs(latent_tail) >= tail_freeze
        )
    return energy_freeze or latent_freeze


def uncertainty_scale(cfg, model_summary: Optional[Mapping[str, object]]) -> float:
    """Return uncertainty-aware damping scale in [0, 1]."""
    if model_summary is None:
        return 1.0
    uncertainty = _metric(model_summary, "uncertainty", "sigma", "std", "variance")
    if uncertainty is None:
        return 1.0
    ref = float(getattr(cfg, "uncertainty_ref", _UNCERTAINTY_LOW))
    power = float(getattr(cfg, "uncertainty_damp_power", 1.0))
    scale = (ref / max(abs(uncertainty), 1e-12)) ** power
    return _clamp(scale, 0.0, 1.0)


def multi_objective_alpha(
    cfg,
    metrics: Mapping[str, object],
    model_summary: Optional[Mapping[str, object]],
) -> float:
    """
    Compute update magnitude alpha in [0,1] based on multiple objectives:
      - gaussian confidence (or conf_ewma if available)
      - exploration score
      - uncertainty scaling
      - change-point penalty
    Deterministic weighted combination.
    """
    w_conf = float(getattr(cfg, "multi_objective_w_conf", 0.55))
    w_explore = float(getattr(cfg, "multi_objective_w_explore", 0.35))
    w_uncert = float(getattr(cfg, "multi_objective_w_uncert", 0.10))
    cp_alpha_multiplier = float(getattr(cfg, "cp_alpha_multiplier", 0.0))

    conf = float(metrics.get(
        "conf_ewma",
        metrics.get("gaussian_confidence", gaussian_confidence(cfg, metrics)),
    ))
    explore = float(metrics.get("explore_score", 0.0))
    uncert_scale = float(uncertainty_scale(cfg, model_summary))

    alpha_raw = w_conf * conf + w_explore * explore + w_uncert * uncert_scale
    if bool(metrics.get("change_point", False)):
        alpha_raw *= cp_alpha_multiplier
    if bool(metrics.get("controller_frozen", False)):
        alpha_raw = 0.0

    alpha = _clamp(alpha_raw, 0.0, 1.0)
    damp_min = float(getattr(cfg, "policy_damp_min", 0.0))
    damp_max = float(getattr(cfg, "policy_damp_max", 1.0))
    if alpha > 0.0:
        alpha = max(damp_min, min(damp_max, alpha))
    return _clamp(alpha, 0.0, 1.0)


def _adjust_k0(
    base: float,
    metrics: Mapping[str, object],
    model_summary: Optional[Mapping[str, object]],
    *,
    kurtosis_high: float,
    kurtosis_good: float,
    skew_good: float,
) -> float:
    kurtosis = _metric(metrics, "kurtosis", "kurtosis_proxy", "excess_kurtosis")
    skew = _metric(metrics, "skew", "skew_proxy", "skewness")
    k0 = base
    if kurtosis is not None and abs(kurtosis) >= kurtosis_high:
        return k0 * _K0_DOWN_FACTOR
    if kurtosis is None or skew is None:
        return k0
    if abs(kurtosis) <= kurtosis_good and abs(skew) <= skew_good:
        if _uncertainty_is_low(model_summary):
            return k0 + _K0_UP_STEP
    return k0

def _apply_k_bounds(
    k0: float,
    vmin: float,
    vmax: float,
    k_min: Optional[float],
    k_max: Optional[float],
    k0_min: float,
    k0_max: float,
) -> float:
    if k_min is None and k_max is None:
        return k0
    span = max(vmax - vmin, _EPS)
    k_val = k0 / span
    if k_min is not None:
        k_val = max(float(k_min), k_val)
    if k_max is not None:
        k_val = min(float(k_max), k_val)
    return _clamp(k_val * span, k0_min, k0_max)


def propose_boost_params(
    cfg,
    cycle_stats: Mapping[str, object],
    last_restart,
    metrics: Mapping[str, object],
    model_summary: Optional[Mapping[str, object]] = None,
) -> BoostParams:
    """
    Deterministic rule-based controller for next-cycle boost parameters.
    """
    k0_min = float(getattr(cfg, "k0_min", 0.0))
    k0_max = float(getattr(cfg, "k0_max", 1.0))
    if k0_min > k0_max:
        raise ValueError(f"k0_min must be <= k0_max, got {(k0_min, k0_max)}")
    base_k0 = float(getattr(cfg, "k0_initial", 0.5))
    k_min = getattr(cfg, "k_min", None)
    k_max = getattr(cfg, "k_max", None)

    kurtosis_high = float(getattr(cfg, "gaussian_excess_kurtosis_high", _KURTOSIS_HIGH))
    kurtosis_good = float(getattr(cfg, "gaussian_excess_kurtosis_good", _KURTOSIS_GOOD))
    skew_good = float(getattr(cfg, "gaussian_skew_good", _SKEW_GOOD))

    k0D_base = base_k0
    k0P_base = base_k0
    if last_restart is not None:
        k0D_base = float(getattr(last_restart, "k0D", base_k0))
        k0P_base = float(getattr(last_restart, "k0P", base_k0))

    vmin_d = _get_value(cycle_stats, "VminD")
    vmax_d = _get_value(cycle_stats, "VmaxD")
    vmin_p = _get_value(cycle_stats, "VminP")
    vmax_p = _get_value(cycle_stats, "VmaxP")
    vavg_d = _get_value(cycle_stats, "VavgD")
    vstd_d = _get_value(cycle_stats, "VstdD")
    vavg_p = _get_value(cycle_stats, "VavgP")
    vstd_p = _get_value(cycle_stats, "VstdP")

    if vmin_d is None or vmax_d is None:
        if last_restart is None:
            raise ValueError("Missing VminD/VmaxD for policy")
        vmin_d = float(getattr(last_restart, "VminD_kJ"))
        vmax_d = float(getattr(last_restart, "VmaxD_kJ"))
    if vmin_p is None or vmax_p is None:
        if last_restart is None:
            raise ValueError("Missing VminP/VmaxP for policy")
        vmin_p = float(getattr(last_restart, "VminP_kJ"))
        vmax_p = float(getattr(last_restart, "VmaxP_kJ"))

    vmin_d, vmax_d = _sanitize_bounds(vmin_d, vmax_d)
    vmin_p, vmax_p = _sanitize_bounds(vmin_p, vmax_p)

    k0D_prop = _adjust_k0(
        k0D_base,
        metrics,
        model_summary,
        kurtosis_high=kurtosis_high,
        kurtosis_good=kurtosis_good,
        skew_good=skew_good,
    )
    k0P_prop = _adjust_k0(
        k0P_base,
        metrics,
        model_summary,
        kurtosis_high=kurtosis_high,
        kurtosis_good=kurtosis_good,
        skew_good=skew_good,
    )
    if not bool(getattr(cfg, "controller_enabled", True)):
        k0D = float(k0D_prop)
        k0P = float(k0P_prop)
    else:
        prev_k0D = k0D_base
        prev_k0P = k0P_base
        change_point = bool(metrics.get("change_point", False))
        if change_point and bool(getattr(cfg, "reset_on_change_point", False)):
            prev_k0D = base_k0
            prev_k0P = base_k0
        if freeze_bias_update(cfg, metrics):
            # Formal stability criterion: freeze if Gaussianity is out of bounds.
            k0D = prev_k0D
            k0P = prev_k0P
        else:
            # Uncertainty-aware damping between previous and proposed values.
            alpha = multi_objective_alpha(cfg, metrics, model_summary)
            k0D = prev_k0D + alpha * (k0D_prop - prev_k0D)
            k0P = prev_k0P + alpha * (k0P_prop - prev_k0P)

    k0D = _clamp(k0D, k0_min, k0_max)
    k0P = _clamp(k0P, k0_min, k0_max)

    def _apply_sigma0_limit(
        k0: float,
        vmin: float,
        vmax: float,
        vavg: Optional[float],
        vstd: Optional[float],
        sigma0: Optional[float],
    ) -> float:
        if sigma0 is None or sigma0 <= 0.0 or vavg is None or vstd is None or vstd <= 0.0:
            return k0
        span = max(vmax - vmin, _EPS)
        denom = max((vmax - vavg) * vstd, _EPS)
        k0_sigma = float(sigma0) * span / denom
        k0 = min(k0, k0_sigma)
        k0 = _clamp(k0, k0_min, k0_max)
        k_val = k0 / span
        if k_min is not None:
            k_val = max(float(k_min), k_val)
        if k_max is not None:
            k_val = min(float(k_max), k_val)
        return _clamp(k_val * span, k0_min, k0_max)

    k0D = _apply_sigma0_limit(k0D, vmin_d, vmax_d, vavg_d, vstd_d, getattr(cfg, "sigma0D", None))
    k0P = _apply_sigma0_limit(k0P, vmin_p, vmax_p, vavg_p, vstd_p, getattr(cfg, "sigma0P", None))
    k0D = _apply_k_bounds(k0D, vmin_d, vmax_d, k_min, k_max, k0_min, k0_max)
    k0P = _apply_k_bounds(k0P, vmin_p, vmax_p, k_min, k_max, k0_min, k0_max)

    deltaV_std = _metric(metrics, "deltaV_std")
    default_deltaV_std_max = None
    if bool(getattr(cfg, "controller_enabled", True)):
        # Default to dual-boost threshold when mode is unspecified in config.
        boost_mode = _infer_boost_mode(cfg)
        if boost_mode == "dihedral":
            default_deltaV_std_max = 6.0
        elif boost_mode == "dual":
            default_deltaV_std_max = 10.0
        else:
            default_deltaV_std_max = 10.0
    deltaV_std_max = _get_value(
        cfg,
        "deltaV_std_max",
        default=default_deltaV_std_max,
        required=bool(getattr(cfg, "controller_enabled", True)),
    )
    deltaV_damp_factor = float(getattr(cfg, "deltaV_damp_factor", 0.5))
    if deltaV_std_max is not None and deltaV_std is not None and deltaV_std > float(deltaV_std_max):
        k0D = _clamp(k0D * deltaV_damp_factor, k0_min, k0_max)
        k0P = _clamp(k0P * deltaV_damp_factor, k0_min, k0_max)

    return BoostParams(
        VminD=vmin_d,
        VmaxD=vmax_d,
        VminP=vmin_p,
        VmaxP=vmax_p,
        k0D=k0D,
        k0P=k0P,
        refED_factor=float(getattr(cfg, "refED_factor", 0.05)),
        refEP_factor=float(getattr(cfg, "refEP_factor", 0.05)),
    )
