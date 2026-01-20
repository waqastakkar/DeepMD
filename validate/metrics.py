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

def write_report_json(path: str | Path, **metrics) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return p
