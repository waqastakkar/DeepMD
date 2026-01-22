"""CLI entrypoint for the GaMD GNN pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from paddle.learn.gnn_pipeline import (
    GraphBuildConfig,
    GraphBuilder,
    TrainingConfig,
    TrajectoryWindowDataset,
    save_importance_maps,
    train_gamd_gnn,
    SaliencyAnalyzer,
)


def _load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _build_frames(payload: Dict[str, np.ndarray], cfg: GraphBuildConfig):
    builder = GraphBuilder(cfg)
    node_features = payload["node_features"]
    positions = payload["positions"]
    contacts = payload.get("contacts")
    hbonds = payload.get("hbonds")
    salts = payload.get("salts")
    covars = payload.get("covariance")
    global_features = payload.get("global_features")

    frames = []
    for i in range(node_features.shape[0]):
        frame = builder.build_frame(
            node_features=node_features[i],
            positions=positions[i],
            contact_mask=None if contacts is None else contacts[i],
            hbond_mask=None if hbonds is None else hbonds[i],
            salt_mask=None if salts is None else salts[i],
            covariance_mask=None if covars is None else covars[i],
            global_features=None if global_features is None else global_features[i],
        )
        frames.append(frame)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description="GaMD GNN pipeline")
    ap.add_argument("--npz", required=True, help="Input NPZ with node features + positions")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--sequence", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=8)
    args = ap.parse_args()

    payload = _load_npz(args.npz)
    frames = _build_frames(payload, GraphBuildConfig())

    labels = {}
    for key in ("delta_v", "state", "rmsd", "rg", "latent"):
        if key in payload:
            labels[key] = payload[key]

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        sequence_len=args.sequence,
    )

    model = train_gamd_gnn(
        frames,
        labels,
        cfg,
        out_dir=args.out,
        latent_dim=args.latent_dim,
        state_classes=int(payload.get("state_classes", 4)),
    )

    analyzer = SaliencyAnalyzer(model)
    window_dataset = TrajectoryWindowDataset(
        frames,
        labels,
        sequence_len=args.sequence,
        batch_size=1,
    )
    batch, _ = next(iter(window_dataset))
    grads = analyzer.gradient_attribution(batch)
    residue_scores = np.mean(np.abs(grads), axis=-1)
    residue_ids = np.arange(len(residue_scores))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_importance_maps(residue_ids, residue_scores, out_dir / "importance.csv", "gradient")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
