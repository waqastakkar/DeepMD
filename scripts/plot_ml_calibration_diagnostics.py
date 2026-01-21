"""
Generate ML calibration diagnostics figure (SVG).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_value(primary: dict, secondary: dict, key: str):
    if key in primary:
        return primary.get(key)
    return secondary.get(key)


def _format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def _extract_array(payload: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    return None


def _load_predictions(run_dir: Path) -> dict:
    npz_path = run_dir / "predictions_test.npz"
    if npz_path.exists():
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
            return data
        except Exception:
            return {}

    json_path = run_dir / "predictions_test.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, list):
            if not payload:
                return {}
            data = {k: [row.get(k) for row in payload] for k in payload[0].keys()}
            return data
        if isinstance(payload, dict):
            return payload

    csv_path = run_dir / "predictions_test.csv"
    if csv_path.exists():
        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        except Exception:
            return {}
        if data.size == 0:
            return {}
        return {name: data[name] for name in data.dtype.names or []}

    return {}


def _normalize_samples(y_true, mu, sigma):
    if y_true is None or mu is None or sigma is None:
        return None, None, None
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    return y_true, mu, sigma


def _compute_scores(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-12):
    err = np.abs(y_true - mu)
    denom = sigma + eps
    ratio = err / denom
    if ratio.ndim > 1:
        return np.max(ratio, axis=1)
    return ratio


def _apply_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot ML calibration diagnostics")
    ap.add_argument("--run", required=True, help="Run directory containing metrics.json")
    ap.add_argument("--out", default=None, help="Output SVG path")
    ap.add_argument("--log", action="store_true", help="Use log scale for scatter panel")
    ap.add_argument("--sample", type=int, default=5000, help="Subsample scatter points")
    ap.add_argument("--alpha", type=float, default=None, help="Override conformal alpha")
    ns = ap.parse_args()

    run_dir = Path(ns.run)
    out_path = Path(ns.out) if ns.out else run_dir / "ml_calibration_diagnostics.svg"

    metrics = _load_json(run_dir / "metrics.json")
    summary = _load_json(run_dir / "model_summary.json")

    alpha = ns.alpha if ns.alpha is not None else _get_value(metrics, summary, "conformal_alpha")
    qhat = _get_value(metrics, summary, "conformal_qhat")
    cov_val = _get_value(metrics, summary, "conformal_coverage_val")
    cov_test = _get_value(metrics, summary, "conformal_coverage_test")
    mean_sigma = _get_value(metrics, summary, "mean_sigma_uncalibrated")
    mean_halfwidth = _get_value(metrics, summary, "conformal_mean_halfwidth_test")
    within_1sigma = _get_value(metrics, summary, "within_1sigma")
    y_std = _get_value(metrics, summary, "y_std")

    target = None
    if alpha is not None:
        try:
            target = 1.0 - float(alpha)
        except Exception:
            target = None

    preds = _load_predictions(run_dir)
    y_true = _extract_array(preds, ("y_true", "y", "y_phys"))
    mu = _extract_array(preds, ("mu", "mu_phys", "y_pred"))
    sigma = _extract_array(preds, ("sigma", "sigma_phys", "y_sigma"))
    covered = _extract_array(preds, ("covered", "is_covered"))
    y_true, mu, sigma = _normalize_samples(y_true, mu, sigma)

    has_scatter = y_true is not None and mu is not None and sigma is not None
    has_panel_c = has_scatter and qhat is not None

    _apply_plot_style()

    if has_panel_c:
        fig = plt.figure(figsize=(7.2, 6.8))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_a = fig.add_subplot(grid[0, :])
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[1, 1])
    else:
        fig = plt.figure(figsize=(7.2, 5.0))
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = None

    # Panel A: coverage + uncertainty
    coverage_vals = [cov_val, cov_test]
    coverage_labels = ["Val", "Test"]
    coverage_colors = ["#1b9e77", "#d95f02"]
    x_positions = np.arange(2)
    bar_vals = [float(v) if v is not None else 0.0 for v in coverage_vals]
    ax_a.bar(x_positions, bar_vals, color=coverage_colors, width=0.6)
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(coverage_labels)
    ax_a.set_ylabel("Coverage")
    ax_a.set_title("A  Coverage and uncertainty")

    for idx, val in enumerate(coverage_vals):
        if val is None:
            ax_a.text(idx, 0.02, "NA", ha="center", va="bottom", fontsize=9)

    if target is not None:
        ax_a.axhline(target, color="black", linestyle="--", linewidth=1.0, label="Target")

    ax_a.set_ylim(0.0, 1.05)

    ax_a_secondary = ax_a.twinx()
    sigma_vals = [mean_sigma, mean_halfwidth]
    sigma_positions = [0, 1]
    sigma_colors = ["#7570b3", "#e7298a"]
    sigma_labels = ["Mean σ (uncal)", "Conformal half-width"]
    for pos, val, color, label in zip(sigma_positions, sigma_vals, sigma_colors, sigma_labels):
        if val is not None:
            ax_a_secondary.plot([pos], [val], marker="o", color=color, label=label)
    ax_a_secondary.set_ylabel("Uncertainty scale")

    handles, labels = ax_a.get_legend_handles_labels()
    handles2, labels2 = ax_a_secondary.get_legend_handles_labels()
    if handles or handles2:
        ax_a.legend(handles + handles2, labels + labels2, loc="upper right", frameon=False)

    # Panel B: residual vs sigma scatter
    ax_b.set_title("B  Residual vs predicted σ")
    if has_scatter:
        rng = np.random.default_rng(0)
        err = np.abs(y_true - mu)
        if err.ndim > 1:
            err = np.max(err, axis=1)
        sigma_vals = sigma
        if sigma_vals.ndim > 1:
            sigma_vals = np.max(sigma_vals, axis=1)

        mask = np.isfinite(err) & np.isfinite(sigma_vals)
        err = err[mask]
        sigma_vals = sigma_vals[mask]
        if ns.log:
            mask_log = (err > 0) & (sigma_vals > 0)
            err = err[mask_log]
            sigma_vals = sigma_vals[mask_log]
            ax_b.set_xscale("log")
            ax_b.set_yscale("log")

        if err.size > ns.sample:
            idx = rng.choice(err.size, size=ns.sample, replace=False)
            err = err[idx]
            sigma_vals = sigma_vals[idx]

        ax_b.scatter(sigma_vals, err, s=8, color="gray", alpha=0.35, edgecolors="none")

        if err.size > 0:
            min_val = np.min([err.min(), sigma_vals.min()])
            max_val = np.max([err.max(), sigma_vals.max()])
            if ns.log:
                min_val = max(min_val, 1e-12)
            ax_b.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
        ax_b.set_xlabel("Predicted σ")
        ax_b.set_ylabel("|Residual|")
    else:
        ax_b.text(0.5, 0.5, "Per-sample predictions not found; scatter omitted.",
                  ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_axis_off()

    # Panel C: within-interval diagnostics
    if ax_c is not None and has_panel_c:
        ax_c.set_title("C  Within-interval diagnostics")
        scores = _compute_scores(y_true, mu, sigma)
        if scores.size == 0:
            ax_c.text(0.5, 0.5, "No per-sample scores available.",
                      ha="center", va="center", transform=ax_c.transAxes)
            ax_c.set_axis_off()
        else:
            ax_c.hist(scores, bins=30, color="#1b9e77", alpha=0.7)
            if qhat is not None:
                ax_c.axvline(float(qhat), color="black", linestyle="--", linewidth=1.0)
            ax_c.set_xlabel("Score = max(|err|/σ)")
            ax_c.set_ylabel("Count")
        if covered is None:
            ax_c.text(0.02, 0.95, "covered unavailable", transform=ax_c.transAxes,
                      ha="left", va="top", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", format="svg")

    print(
        "alpha={alpha} target={target} cov_val={cov_val} cov_test={cov_test} qhat={qhat} "
        "mean_sigma={mean_sigma} mean_halfwidth={mean_halfwidth} within_1sigma={within_1sigma}".format(
            alpha=_format_value(alpha),
            target=_format_value(target),
            cov_val=_format_value(cov_val),
            cov_test=_format_value(cov_test),
            qhat=_format_value(qhat),
            mean_sigma=_format_value(mean_sigma),
            mean_halfwidth=_format_value(mean_halfwidth),
            within_1sigma=_format_value(within_1sigma),
        )
    )
    if y_std:
        try:
            mean_sigma_std = float(mean_sigma) / float(y_std)
            mean_halfwidth_std = float(mean_halfwidth) / float(y_std)
        except Exception:
            mean_sigma_std = None
            mean_halfwidth_std = None
        print(
            "mean_sigma/y_std={sigma_std} mean_halfwidth/y_std={halfwidth_std}".format(
                sigma_std=_format_value(mean_sigma_std),
                halfwidth_std=_format_value(mean_halfwidth_std),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
