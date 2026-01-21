"""
validate/pmf.py — Estimate 1D PMF from an energy-like series (demo)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import json

def pmf_from_series(series: np.ndarray, bins: int = 50):
    hist, edges = np.histogram(series, bins=bins, density=True)
    hist = np.maximum(hist, 1e-12)
    pmf = -np.log(hist)
    centers = 0.5*(edges[:-1] + edges[1:])
    pmf -= np.min(pmf)
    return centers, pmf

def save_pmf_json(path: str | Path, x: np.ndarray, pmf: np.ndarray) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x": x.tolist(), "pmf": pmf.tolist()}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
