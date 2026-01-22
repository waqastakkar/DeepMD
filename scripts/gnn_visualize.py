"""Summarize residue importance and network edges for publication workflows."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize GaMD GNN outputs")
    ap.add_argument("--importance", required=True, help="CSV from save_importance_maps")
    ap.add_argument("--edges", help="Optional edge list CSV")
    ap.add_argument("--out", required=True, help="Output report text")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    importance = pd.read_csv(args.importance)
    top = importance.sort_values("importance", ascending=False).head(args.top)

    lines = [
        "GaMD GNN Summary",
        "================",
        f"Top {args.top} residues by importance:",
    ]
    for _, row in top.iterrows():
        lines.append(f"  Residue {int(row['residue'])}: {row['importance']:.4f}")

    if args.edges:
        edges = pd.read_csv(args.edges)
        lines.append("")
        lines.append("Edge weight summary:")
        lines.append(f"  Edges: {len(edges)}")
        lines.append(f"  Mean weight: {edges['weight'].mean():.4f}")
        lines.append(f"  Max weight: {edges['weight'].max():.4f}")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
