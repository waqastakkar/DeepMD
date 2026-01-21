"""
validate/metrics.py — Simple post-run metrics helpers
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

def anharmonicity_gamma(data: np.ndarray) -> float:
    # excess kurtosis as a proxy (0 for Gaussian). Use absolute value.
    m2 = np.mean((data - data.mean())**2) + 1e-12
    m4 = np.mean((data - data.mean())**4)
    excess = m4 / (m2*m2) - 3.0
    return float(abs(excess))

def gaussianity_report(data: np.ndarray) -> dict[str, float]:
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return {"skewness": 0.0, "excess_kurtosis": 0.0, "tail_risk": 0.0}
    centered = x - x.mean()
    m2 = np.mean(centered**2) + 1e-12
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    skewness = m3 / (m2**1.5)
    excess_kurtosis = m4 / (m2*m2) - 3.0
    z = centered / np.sqrt(m2)
    tail_risk = np.mean(np.abs(z) > 3.0)
    return {
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
        "tail_risk": float(tail_risk),
    }

def aggregate_gaussianity(reports: list[dict[str, float]]) -> dict[str, float]:
    """
    Combine per-dimension gaussianity reports conservatively.
    """
    if not reports:
        return {"skewness": 0.0, "excess_kurtosis": 0.0, "tail_risk": 0.0}
    skewness = max(abs(r.get("skewness", 0.0)) for r in reports)
    excess_kurtosis = max(abs(r.get("excess_kurtosis", 0.0)) for r in reports)
    tail_risk = max(r.get("tail_risk", 0.0) for r in reports)
    return {
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
        "tail_risk": float(tail_risk),
    }

def write_report_json(path: str | Path, **metrics) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return p

def detect_change_point(
    values: np.ndarray,
    window: int = 5,
    z_threshold: float = 3.0,
) -> dict[str, float | bool]:
    """
    Deterministic change-point detector using a windowed z-score on the last point.

    Given a 1D series values of per-cycle scalars (e.g., Etot mean per cycle),
    compute mean/std over the previous `window` points and compute z-score of the last point.
    If |z| >= z_threshold, report change_point=True.
    """
    x = np.asarray(values, dtype=float).ravel()
    window = int(window)
    z_threshold = float(z_threshold)
    if x.size < window + 1:
        return {
            "change_point": False,
            "z_score": 0.0,
            "mean_ref": 0.0,
            "std_ref": 0.0,
            "window": window,
            "z_threshold": z_threshold,
        }
    ref = x[-(window + 1):-1]
    mean_ref = float(np.mean(ref))
    std_ref = float(np.std(ref))
    std_ref = max(std_ref, 1e-12)
    z_score = float((x[-1] - mean_ref) / std_ref)
    return {
        "change_point": abs(z_score) >= z_threshold,
        "z_score": z_score,
        "mean_ref": mean_ref,
        "std_ref": std_ref,
        "window": window,
        "z_threshold": z_threshold,
    }
