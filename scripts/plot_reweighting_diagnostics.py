#!/usr/bin/env python3
"""Plot GaMD reweighting diagnostics from bias_plan_cycle_*.json files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt


CYCLE_PATTERN = re.compile(r"bias_plan_cycle_(\d+)\.json")

ESS_COLOR = "#1b9e77"
ENTROPY_COLOR = "#d95f02"
WCV_COLOR = "#7570b3"
FREEZE_COLOR = "#d9d9d9"
NOT_OK_COLOR = "#e7298a"


@dataclass
class CycleMetrics:
    cycle: int
    ess_frac: float
    entropy_norm: float
    w_cv: Optional[float]
    reweight_ok: Optional[bool]
    freeze: bool
    ess_min: Optional[float]
    entropy_min: Optional[float]


def parse_cycle_from_name(path: Path) -> Optional[int]:
    match = CYCLE_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _get_nested(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_reweight_ok(data: Dict[str, Any]) -> Optional[bool]:
    metrics = data.get("metrics", {})
    controller = data.get("controller", {})
    for candidate in (
        _get_nested(metrics, "reweight", "reweight_ok"),
        metrics.get("reweight_ok"),
        controller.get("reweight_ok"),
    ):
        if candidate is not None:
            return bool(candidate)
    return None


def extract_thresholds(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    metrics = data.get("metrics", {})
    reweight = metrics.get("reweight", {}) if isinstance(metrics, dict) else {}
    ess_min = reweight.get("ess_min") if isinstance(reweight, dict) else None
    entropy_min = reweight.get("entropy_min") if isinstance(reweight, dict) else None
    return ess_min, entropy_min


def load_cycle_metrics(path: Path) -> Optional[CycleMetrics]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = data.get("metrics", {})
    reweight = metrics.get("reweight") if isinstance(metrics, dict) else None
    if not isinstance(reweight, dict):
        return None

    ess_frac = reweight.get("ess_frac")
    entropy_norm = reweight.get("entropy_norm")
    if ess_frac is None or entropy_norm is None:
        return None

    cycle = data.get("cycle")
    if cycle is None:
        cycle = parse_cycle_from_name(path)
    if cycle is None:
        return None

    w_cv = reweight.get("w_cv")
    reweight_ok = extract_reweight_ok(data)
    controller = data.get("controller", {})
    freeze = bool(
        controller.get("freeze_bias_update") or controller.get("controller_frozen")
    )
    ess_min, entropy_min = extract_thresholds(data)

    return CycleMetrics(
        cycle=int(cycle),
        ess_frac=float(ess_frac),
        entropy_norm=float(entropy_norm),
        w_cv=float(w_cv) if w_cv is not None else None,
        reweight_ok=reweight_ok,
        freeze=freeze,
        ess_min=ess_min,
        entropy_min=entropy_min,
    )


def scan_cycles(outdir: Path) -> List[CycleMetrics]:
    metrics: List[CycleMetrics] = []
    for path in sorted(outdir.glob("bias_plan_cycle_*.json")):
        item = load_cycle_metrics(path)
        if item is not None:
            metrics.append(item)
    return metrics


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "legend.frameon": False,
        }
    )


def _apply_bold_ticks(ax: plt.Axes) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def _axvspan_freeze(ax: plt.Axes, cycles: Iterable[int]) -> None:
    for cycle in cycles:
        ax.axvspan(cycle - 0.5, cycle + 0.5, color=FREEZE_COLOR, alpha=0.15, zorder=0)


def plot_diagnostics(metrics: List[CycleMetrics], outpath: Path) -> None:
    if len(metrics) < 2:
        print("Not enough reweighting data points to plot (need at least 2).")
        return

    metrics_sorted = sorted(metrics, key=lambda item: item.cycle)
    cycles = [item.cycle for item in metrics_sorted]
    ess_vals = [item.ess_frac for item in metrics_sorted]
    entropy_vals = [item.entropy_norm for item in metrics_sorted]
    has_w_cv = any(item.w_cv is not None for item in metrics_sorted)

    freeze_cycles = [item.cycle for item in metrics_sorted if item.freeze]
    not_ok_cycles = [
        item.cycle for item in metrics_sorted if item.reweight_ok is False
    ]

    ess_min = next((item.ess_min for item in metrics_sorted if item.ess_min is not None), 0.1)
    entropy_min = next(
        (item.entropy_min for item in metrics_sorted if item.entropy_min is not None),
        0.7,
    )

    configure_matplotlib()

    if has_w_cv:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.2, 4.2),
            gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.15},
        )
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(7.2, 4.2))
        ax_bottom = None

    _axvspan_freeze(ax_top, freeze_cycles)
    ax_top.plot(
        cycles,
        ess_vals,
        color=ESS_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="ESS fraction",
    )
    ax_top.plot(
        cycles,
        entropy_vals,
        color=ENTROPY_COLOR,
        marker="s",
        linewidth=2.0,
        markersize=5,
        label="Normalized entropy",
    )

    cycle_to_ess = {item.cycle: item.ess_frac for item in metrics_sorted}
    if not_ok_cycles:
        not_ok_ess = [cycle_to_ess[cycle] for cycle in not_ok_cycles]
        ax_top.scatter(
            not_ok_cycles,
            not_ok_ess,
            marker="*",
            s=90,
            color=NOT_OK_COLOR,
            label="Reweighting not OK",
            zorder=3,
        )

    ax_top.axhline(ess_min, color=ESS_COLOR, linestyle="--", linewidth=1.2, alpha=0.7)
    ax_top.axhline(
        entropy_min, color=ENTROPY_COLOR, linestyle="--", linewidth=1.2, alpha=0.7
    )
    ax_top.set_ylabel("Fraction / Entropy", fontweight="bold")
    ax_top.set_ylim(0.0, 1.05)
    ax_top.set_xlim(min(cycles) - 0.5, max(cycles) + 0.5)
    ax_top.legend(loc="best", prop={"weight": "bold"})
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    _apply_bold_ticks(ax_top)

    if has_w_cv and ax_bottom is not None:
        _axvspan_freeze(ax_bottom, freeze_cycles)
        w_cv_cycles = [item.cycle for item in metrics_sorted if item.w_cv is not None]
        w_cv_vals = [item.w_cv for item in metrics_sorted if item.w_cv is not None]
        ax_bottom.plot(
            w_cv_cycles,
            w_cv_vals,
            color=WCV_COLOR,
            marker="^",
            linewidth=2.0,
            markersize=5,
            label="Weight CV",
        )
        ax_bottom.set_ylabel("Weight CV", fontweight="bold")
        ax_bottom.set_xlabel("Cycle", fontweight="bold")
        ax_bottom.legend(loc="best", prop={"weight": "bold"})
        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)
        _apply_bold_ticks(ax_bottom)
    else:
        ax_top.set_xlabel("Cycle", fontweight="bold")

    fig.savefig(outpath, format="svg", bbox_inches="tight")
    print(f"Wrote {outpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GaMD reweighting diagnostics from bias_plan_cycle JSON files."
    )
    parser.add_argument("--run", required=True, help="Output directory to scan")
    parser.add_argument(
        "--out",
        default=None,
        help="SVG output path (default: <run>/reweighting_diagnostics.svg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.run)
    if not outdir.is_dir():
        raise SystemExit(f"Run directory not found: {outdir}")

    outpath = Path(args.out) if args.out else outdir / "reweighting_diagnostics.svg"
    metrics = scan_cycles(outdir)
    plot_diagnostics(metrics, outpath)


if __name__ == "__main__":
    main()
