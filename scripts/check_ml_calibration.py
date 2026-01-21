#!/usr/bin/env python3
"""Lightweight calibration sanity checks for DBMDX runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ML calibration metrics for a run directory")
    ap.add_argument("run_dir", type=str, help="Path to run directory containing metrics.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"FAIL: metrics.json not found in {run_dir}")
        return 1

    metrics = _load_json(metrics_path)
    errors = []

    alpha = metrics.get("conformal_alpha")
    qhat = metrics.get("conformal_qhat")
    if alpha is None or qhat is None:
        errors.append("Missing conformal_alpha or conformal_qhat in metrics.json")
    else:
        if not np.isfinite(qhat) or qhat <= 0:
            errors.append(f"Invalid conformal_qhat: {qhat}")

    within_1sigma = metrics.get("within_1sigma")
    if within_1sigma is None:
        errors.append("Missing within_1sigma in metrics.json")
    else:
        if not np.isfinite(within_1sigma) or within_1sigma <= 0.0:
            errors.append(f"within_1sigma is not positive: {within_1sigma}")

    mae = metrics.get("mae")
    if mae is None or not np.isfinite(mae):
        errors.append("MAE missing or not finite")

    y_test_std = metrics.get("y_test_std")
    if y_test_std is None:
        summary_path = run_dir / "model_summary.json"
        if summary_path.exists():
            y_test_std = _load_json(summary_path).get("y_test_std")
    if y_test_std is None:
        errors.append("Missing y_test_std in metrics.json/model_summary.json")
    else:
        if mae is not None and np.isfinite(mae):
            if mae >= 0.5 * float(y_test_std):
                errors.append(
                    f"MAE sanity check failed: mae={mae} >= 0.5 * std(y_test)={0.5 * float(y_test_std)}"
                )

    conformal_valid = metrics.get("conformal_valid")
    if conformal_valid:
        coverage_val = metrics.get("conformal_coverage_val")
        if coverage_val is None:
            errors.append("Missing conformal_coverage_val for conformal_valid run")
        else:
            if alpha is None:
                errors.append("Missing conformal_alpha for coverage check")
            else:
                alpha_val = float(alpha)
                coverage_floor = (1.0 - alpha_val) - 0.02
                if float(coverage_val) < coverage_floor:
                    errors.append(
                        f"Conformal coverage check failed: {coverage_val} < {coverage_floor}"
                    )

    if errors:
        print("FAIL: calibration checks failed")
        for err in errors:
            print(f" - {err}")
        return 1

    print("PASS: calibration checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
