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

def write_report_json(path: str | Path, **metrics) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return p
