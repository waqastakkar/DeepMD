#!/usr/bin/env python3
"""Plot controller diagnostics from bias_plan_cycle_*.json files.

This script scans a run directory for bias_plan_cycle_*.json files and
produces a two-panel SVG figure: Gaussianity metrics and k0 evolution.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt


CYCLE_RE = re.compile(r"bias_plan_cycle_(\d+)")

COLORS = {
    "skew": "#1b9e77",
    "kurt": "#d95f02",
    "tail": "#7570b3",
    "k0D": "#e7298a",
    "k0P": "#66a61e",
    "freeze": "#d9d9d9",
    "confidence": "#4d4d4d",
}


def parse_cycle(path: Path) -> int | None:
    match = CYCLE_RE.search(path.stem)
    if match:
        return int(match.group(1))
    return None


def load_bias_plan(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_cycles(run_dir: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    thresholds: dict[str, float] = {}
    for path in sorted(run_dir.glob("bias_plan_cycle_*.json")):
        try:
            data = load_bias_plan(path)
        except json.JSONDecodeError as exc:
            warnings.warn(f"Skipping {path.name}: invalid JSON ({exc})")
            continue

        cycle = parse_cycle(path)
        if cycle is None:
            cycle = data.get("cycle")
        if cycle is None:
            warnings.warn(f"Skipping {path.name}: missing cycle index")
            continue

        metrics = data.get("metrics", {})
        skew = metrics.get("skewness")
        kurt = metrics.get("excess_kurtosis")
        tail = metrics.get("tail_risk")
        if skew is None or kurt is None or tail is None:
            warnings.warn(f"Skipping cycle {cycle}: missing gaussianity metrics")
            continue

        params = data.get("params", {})
        k0d = params.get("k0D")
        k0p = params.get("k0P")
        if k0d is None or k0p is None:
            warnings.warn(f"Skipping cycle {cycle}: missing k0D/k0P")
            continue

        controller = data.get("controller", {})
        freeze = bool(controller.get("freeze_bias_update", False))
        confidence = controller.get("gaussian_confidence")

        config = data.get("config", {})
        if config and not thresholds:
            for key in (
                "gaussian_skew_good",
                "gaussian_excess_kurtosis_good",
                "gaussian_tail_risk_good",
            ):
                if key in config:
                    thresholds[key] = config[key]

        records.append(
            {
                "cycle": int(cycle),
                "skew": abs(float(skew)),
                "kurt": abs(float(kurt)),
                "tail": float(tail),
                "k0D": float(k0d),
                "k0P": float(k0p),
                "freeze": freeze,
                "confidence": None if confidence is None else float(confidence),
            }
        )

    records.sort(key=lambda item: item["cycle"])
    return records, thresholds


def apply_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(width=1.2, labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot controller diagnostics from bias_plan_cycle_*.json files."
    )
    parser.add_argument("--run", required=True, help="Run directory with JSON files")
    parser.add_argument(
        "--out",
        default=None,
        help="Output SVG path (default: <run>/controller_diagnostics.svg)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    records, thresholds = gather_cycles(run_dir)
    if not records:
        raise SystemExit("No valid bias_plan_cycle_*.json files found")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else run_dir / "controller_diagnostics.svg"
    )

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
        }
    )

    cycles = [item["cycle"] for item in records]
    skew_vals = [item["skew"] for item in records]
    kurt_vals = [item["kurt"] for item in records]
    tail_vals = [item["tail"] for item in records]
    k0d_vals = [item["k0D"] for item in records]
    k0p_vals = [item["k0P"] for item in records]
    freeze_cycles = [item["cycle"] for item in records if item["freeze"]]

    fig, (ax_metrics, ax_k0) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )

    for cycle in freeze_cycles:
        ax_metrics.axvspan(
            cycle - 0.5,
            cycle + 0.5,
            color=COLORS["freeze"],
            alpha=0.15,
            zorder=0,
        )

    ax_metrics.plot(cycles, skew_vals, color=COLORS["skew"], linewidth=2.0, label="|skewness|")
    ax_metrics.plot(
        cycles, kurt_vals, color=COLORS["kurt"], linewidth=2.0, label="|excess kurtosis|"
    )
    ax_metrics.plot(cycles, tail_vals, color=COLORS["tail"], linewidth=2.0, label="tail risk")

    if "gaussian_skew_good" in thresholds:
        ax_metrics.axhline(
            thresholds["gaussian_skew_good"],
            color=COLORS["skew"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
    if "gaussian_excess_kurtosis_good" in thresholds:
        ax_metrics.axhline(
            thresholds["gaussian_excess_kurtosis_good"],
            color=COLORS["kurt"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )
    if "gaussian_tail_risk_good" in thresholds:
        ax_metrics.axhline(
            thresholds["gaussian_tail_risk_good"],
            color=COLORS["tail"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )

    ax_metrics.set_ylabel("Gaussianity metric", fontsize=12, fontweight="bold")
    ax_metrics.set_title("Gaussianity metrics vs cycle", fontsize=14, fontweight="bold")
    ax_metrics.legend(frameon=False)
    apply_axis_style(ax_metrics)

    ax_k0.plot(cycles, k0d_vals, color=COLORS["k0D"], linewidth=2.0, label="k0D")
    ax_k0.plot(cycles, k0p_vals, color=COLORS["k0P"], linewidth=2.0, label="k0P")

    if freeze_cycles:
        freeze_indices = [i for i, item in enumerate(records) if item["freeze"]]
        ax_k0.scatter(
            [cycles[i] for i in freeze_indices],
            [k0d_vals[i] for i in freeze_indices],
            color=COLORS["k0D"],
            marker="x",
            s=50,
            linewidths=1.5,
        )
        ax_k0.scatter(
            [cycles[i] for i in freeze_indices],
            [k0p_vals[i] for i in freeze_indices],
            color=COLORS["k0P"],
            marker="x",
            s=50,
            linewidths=1.5,
        )

    confidences = [item["confidence"] for item in records]
    if any(value is not None for value in confidences):
        ax_conf = ax_k0.twinx()
        conf_vals = [value if value is not None else float("nan") for value in confidences]
        ax_conf.plot(
            cycles,
            conf_vals,
            color=COLORS["confidence"],
            linewidth=1.2,
            alpha=0.5,
            label="gaussian confidence",
        )
        ax_conf.set_ylabel("Confidence", fontsize=12, fontweight="bold")
        apply_axis_style(ax_conf)

    ax_k0.set_xlabel("Cycle", fontsize=12, fontweight="bold")
    ax_k0.set_ylabel("k0", fontsize=12, fontweight="bold")
    ax_k0.set_title("k0 evolution vs cycle", fontsize=14, fontweight="bold")
    ax_k0.legend(frameon=False, loc="upper left")
    apply_axis_style(ax_k0)

    fig.savefig(out_path, format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
