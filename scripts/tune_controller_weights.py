#!/usr/bin/env python3
"""Tune controller weights on a benchmark run directory.

This script performs a deterministic, development-time random search over
controller weight settings and writes the best candidate to a YAML fragment
that can be copied into a config.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


CYCLE_PATTERN = re.compile(r"bias_plan_cycle_(\d+)\.json$")


@dataclass(frozen=True)
class CycleMetrics:
    gaussian_confidence: float
    frozen: bool
    change_point: bool
    explore_score: float
    uncertainty_scale: float
    alpha_proxy: float


@dataclass(frozen=True)
class Candidate:
    w_conf: float
    w_explore: float
    w_uncert: float
    cp_alpha_multiplier: float


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    exploration_reward: float
    stability_penalty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-search tuner for controller weights using benchmark "
            "bias_plan_cycle_*.json summaries."
        )
    )
    parser.add_argument("--run", required=True, help="Benchmark run directory")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to output YAML fragment (default: "
            "<run>/tuned_controller_weights.yaml)"
        ),
    )
    parser.add_argument("--budget", type=int, default=40, help="Trial budget")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def _cycle_index(path: Path) -> int:
    match = CYCLE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Unexpected cycle filename: {path}")
    return int(match.group(1))


def load_cycle_metrics(run_dir: Path) -> List[CycleMetrics]:
    cycle_paths = sorted(
        run_dir.glob("bias_plan_cycle_*.json"),
        key=_cycle_index,
    )
    if not cycle_paths:
        raise FileNotFoundError(
            f"No bias_plan_cycle_*.json files found in {run_dir}"
        )

    metrics: List[CycleMetrics] = []
    for path in cycle_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        controller = payload.get("controller", {})
        gaussian_confidence = float(controller.get("gaussian_confidence", 0.0))
        frozen = bool(
            controller.get("freeze_bias_update")
            or controller.get("controller_frozen")
        )
        change_point = bool(controller.get("change_point"))
        explore_score = float(controller.get("explore_score", 0.0))
        uncertainty_scale = float(controller.get("uncertainty_scale", 1.0))

        if "multi_objective_alpha" in payload:
            alpha_proxy = float(payload["multi_objective_alpha"])
        else:
            # Proxy intended for diagnostics when alpha is not logged.
            alpha_proxy = gaussian_confidence - uncertainty_scale

        metrics.append(
            CycleMetrics(
                gaussian_confidence=gaussian_confidence,
                frozen=frozen,
                change_point=change_point,
                explore_score=explore_score,
                uncertainty_scale=uncertainty_scale,
                alpha_proxy=alpha_proxy,
            )
        )

    return metrics


def latin_hypercube_samples(
    rng: np.random.Generator,
    budget: int,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Generate a Latin-hypercube-like sample matrix."""
    dims = len(bounds)
    samples = np.zeros((budget, dims), dtype=float)
    for dim, (low, high) in enumerate(bounds):
        perm = rng.permutation(budget)
        offsets = rng.random(budget)
        values = (perm + offsets) / budget
        samples[:, dim] = low + values * (high - low)
    return samples


def normalize_weights(w_conf: float, w_explore: float, w_uncert: float) -> Tuple[float, float, float]:
    total = w_conf + w_explore + w_uncert
    if total <= 0:
        raise ValueError("Weight sum must be positive")
    return w_conf / total, w_explore / total, w_uncert / total


def evaluate_candidate(candidate: Candidate, metrics: Sequence[CycleMetrics]) -> ScoredCandidate:
    """Compute the scalar objective.

    Objective definition:
      - penalty_freeze = (# frozen cycles) / (# cycles)
      - penalty_cp = (# change points) / (# cycles)
      - penalty_gauss = mean(max(0, 0.8 - gaussian_confidence))
      - stability_penalty = 1.5*penalty_freeze + 2.0*penalty_cp + 1.0*penalty_gauss
      - exploration_reward = mean(explore_score)
      - score = exploration_reward - stability_penalty

    The exploration score is taken directly from the logged controller metrics
    (0 if missing), ensuring deterministic, data-driven evaluation.
    """
    if not metrics:
        raise ValueError("No cycle metrics provided")

    frozen_count = sum(1 for item in metrics if item.frozen)
    cp_count = sum(1 for item in metrics if item.change_point)
    gaussian_penalties = [max(0.0, 0.8 - item.gaussian_confidence) for item in metrics]

    penalty_freeze = frozen_count / len(metrics)
    penalty_cp = cp_count / len(metrics)
    penalty_gauss = float(np.mean(gaussian_penalties))
    stability_penalty = 1.5 * penalty_freeze + 2.0 * penalty_cp + 1.0 * penalty_gauss

    exploration_reward = float(np.mean([item.explore_score for item in metrics]))
    score = exploration_reward - stability_penalty

    return ScoredCandidate(
        candidate=candidate,
        score=score,
        exploration_reward=exploration_reward,
        stability_penalty=stability_penalty,
    )


def tune_weights(
    metrics: Sequence[CycleMetrics],
    budget: int,
    seed: int,
) -> Tuple[ScoredCandidate, List[ScoredCandidate]]:
    rng = np.random.default_rng(seed)
    bounds = [
        (0.3, 0.8),  # w_conf
        (0.1, 0.6),  # w_explore
        (0.0, 0.3),  # w_uncert
        (0.0, 0.5),  # cp_alpha_multiplier
    ]
    raw_samples = latin_hypercube_samples(rng, budget, bounds)

    scored: List[ScoredCandidate] = []
    for sample in raw_samples:
        w_conf, w_explore, w_uncert, cp_alpha_multiplier = sample.tolist()
        w_conf, w_explore, w_uncert = normalize_weights(w_conf, w_explore, w_uncert)
        candidate = Candidate(
            w_conf=w_conf,
            w_explore=w_explore,
            w_uncert=w_uncert,
            cp_alpha_multiplier=cp_alpha_multiplier,
        )
        scored.append(evaluate_candidate(candidate, metrics))

    scored.sort(key=lambda item: item.score, reverse=True)
    best = scored[0]
    return best, scored[:5]


def format_yaml(best: ScoredCandidate, budget: int, seed: int) -> str:
    candidate = best.candidate
    return (
        "controller:\n"
        f"  w_conf: {candidate.w_conf:.6f}\n"
        f"  w_explore: {candidate.w_explore:.6f}\n"
        f"  w_uncert: {candidate.w_uncert:.6f}\n"
        f"  cp_alpha_multiplier: {candidate.cp_alpha_multiplier:.6f}\n"
        "tuning:\n"
        "  objective: exploration_reward - stability_penalty\n"
        f"  best_score: {best.score:.6f}\n"
        f"  budget: {budget}\n"
        f"  seed: {seed}\n"
    )


def print_summary(best: ScoredCandidate, top: Iterable[ScoredCandidate]) -> None:
    candidate = best.candidate
    print("Best candidate:")
    print(
        "  w_conf={:.4f} w_explore={:.4f} w_uncert={:.4f} "
        "cp_alpha_multiplier={:.4f}".format(
            candidate.w_conf,
            candidate.w_explore,
            candidate.w_uncert,
            candidate.cp_alpha_multiplier,
        )
    )
    print(
        "  score={:.4f} exploration_reward={:.4f} stability_penalty={:.4f}".format(
            best.score,
            best.exploration_reward,
            best.stability_penalty,
        )
    )
    print("Top 5 candidates:")
    for rank, entry in enumerate(top, start=1):
        cand = entry.candidate
        print(
            "  #{rank}: score={score:.4f} w_conf={w_conf:.4f} "
            "w_explore={w_explore:.4f} w_uncert={w_uncert:.4f} "
            "cp_alpha_multiplier={cp_alpha_multiplier:.4f}".format(
                rank=rank,
                score=entry.score,
                w_conf=cand.w_conf,
                w_explore=cand.w_explore,
                w_uncert=cand.w_uncert,
                cp_alpha_multiplier=cand.cp_alpha_multiplier,
            )
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    output_path = Path(args.out) if args.out else run_dir / "tuned_controller_weights.yaml"

    metrics = load_cycle_metrics(run_dir)
    best, top = tune_weights(metrics, args.budget, args.seed)

    output_path.write_text(format_yaml(best, args.budget, args.seed), encoding="utf-8")
    print_summary(best, top)
    print(f"Wrote tuned weights to {output_path}")


if __name__ == "__main__":
    main()
