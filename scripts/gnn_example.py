"""Generate a minimal GaMD GNN example dataset (alanine + 100-residue protein)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _synthetic_features(num_frames: int, num_residues: int, rng: np.random.Generator):
    node_features = rng.normal(size=(num_frames, num_residues, 12)).astype(np.float32)
    positions = rng.normal(size=(num_frames, num_residues, 3)).astype(np.float32)
    contacts = rng.random(size=(num_frames, num_residues, num_residues)) > 0.8
    contacts = contacts.astype(np.float32)

    delta_v = rng.normal(size=(num_frames, 1)).astype(np.float32)
    rmsd = np.abs(rng.normal(size=(num_frames, 1))).astype(np.float32)
    rg = np.abs(rng.normal(size=(num_frames, 1))).astype(np.float32)
    state = rng.integers(0, 3, size=(num_frames, 1)).astype(np.int32)
    latent = rng.normal(size=(num_frames, 8)).astype(np.float32)

    return {
        "node_features": node_features,
        "positions": positions,
        "contacts": contacts,
        "delta_v": delta_v,
        "rmsd": rmsd,
        "rg": rg,
        "state": state,
        "latent": latent,
        "state_classes": np.array([3], dtype=np.int32),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate GaMD GNN example inputs")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(2025)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alanine = _synthetic_features(args.frames, 2, rng)
    protein = _synthetic_features(args.frames, 100, rng)

    np.savez_compressed(out_dir / "alanine_gamd.npz", **alanine)
    np.savez_compressed(out_dir / "protein_gamd.npz", **protein)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
