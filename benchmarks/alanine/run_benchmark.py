"""
Minimal alanine dipeptide-style benchmark (synthetic series).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paddle.validate.metrics import gaussianity_report, write_report_json
from paddle.validate.pmf import pmf_from_series, save_pmf_json


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_list(value: object, length: int, default: list[float]) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        return default
    return [float(x) for x in value]


def generate_series(
    steps: int,
    seed: int,
    centers: list[float],
    scales: list[float],
    weights: list[float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(centers), size=steps, p=weights)
    loc = np.take(centers, choices)
    scale = np.take(scales, choices)
    return rng.normal(loc=loc, scale=scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to benchmark config (json)",
    )
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("out"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)

    steps = int(args.steps or cfg.get("steps", 5000))
    bins = int(args.bins or cfg.get("bins", 60))
    seed = int(args.seed or cfg.get("seed", 2025))

    centers = _coerce_list(cfg.get("mixture_centers"), 2, [-1.1, 1.2])
    scales = _coerce_list(cfg.get("mixture_scales"), 2, [0.35, 0.25])
    weights = _coerce_list(cfg.get("mixture_weights"), 2, [0.6, 0.4])
    total = sum(weights)
    weights = [w / total for w in weights]

    args.outdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    series = generate_series(steps, seed, centers, scales, weights)
    x, pmf = pmf_from_series(series, bins=bins)
    save_pmf_json(args.outdir / "pmf.json", x, pmf)

    gaussianity = gaussianity_report(series)
    write_report_json(args.outdir / "metrics.json", **gaussianity)

    elapsed = time.perf_counter() - start
    runtime = {
        "steps": steps,
        "bins": bins,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed > 0 else None,
    }
    write_report_json(args.outdir / "runtime.json", **runtime)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
