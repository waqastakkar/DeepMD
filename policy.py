"""
policy.py — Rule-based closed-loop bias planning
"""
from __future__ import annotations

from typing import Mapping, Optional

from core.params import BoostParams

_KURTOSIS_HIGH = 1.0
_KURTOSIS_GOOD = 0.2
_SKEW_GOOD = 0.2
_UNCERTAINTY_LOW = 0.2
_K0_DOWN_FACTOR = 0.8
_K0_UP_STEP = 0.05
_MIN_SPAN = 1e-6


def _get_value(source: object, key: str) -> Optional[float]:
    if source is None:
        return None
    if isinstance(source, Mapping):
        if key in source:
            return float(source[key])
        return None
    if hasattr(source, key):
        return float(getattr(source, key))
    return None


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


def _uncertainty_is_low(model_summary: Optional[Mapping[str, object]]) -> bool:
    if model_summary is None:
        return True
    value = _metric(model_summary, "uncertainty", "sigma", "std", "variance")
    if value is None:
        return False
    return abs(value) <= _UNCERTAINTY_LOW


def _adjust_k0(
    base: float,
    metrics: Mapping[str, object],
    model_summary: Optional[Mapping[str, object]],
) -> float:
    kurtosis = _metric(metrics, "kurtosis", "kurtosis_proxy", "excess_kurtosis")
    skew = _metric(metrics, "skew", "skew_proxy")
    k0 = base
    if kurtosis is not None and abs(kurtosis) >= _KURTOSIS_HIGH:
        return k0 * _K0_DOWN_FACTOR
    if kurtosis is None or skew is None:
        return k0
    if abs(kurtosis) <= _KURTOSIS_GOOD and abs(skew) <= _SKEW_GOOD:
        if _uncertainty_is_low(model_summary):
            return k0 + _K0_UP_STEP
    return k0


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

    k0D_base = base_k0
    k0P_base = base_k0
    if last_restart is not None:
        k0D_base = float(getattr(last_restart, "k0D", base_k0))
        k0P_base = float(getattr(last_restart, "k0P", base_k0))

    vmin_d = _get_value(cycle_stats, "VminD")
    vmax_d = _get_value(cycle_stats, "VmaxD")
    vmin_p = _get_value(cycle_stats, "VminP")
    vmax_p = _get_value(cycle_stats, "VmaxP")

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

    k0D = _adjust_k0(k0D_base, metrics, model_summary)
    k0P = _adjust_k0(k0P_base, metrics, model_summary)

    k0D = _clamp(k0D, k0_min, k0_max)
    k0P = _clamp(k0P, k0_min, k0_max)

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
