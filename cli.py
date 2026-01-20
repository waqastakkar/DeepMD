"""
cli.py — One-entry command line for paddle
"""
from __future__ import annotations

import argparse
from pathlib import Path

from paddle.config import SimulationConfig
from paddle.stages.cmd import run_cmd
from paddle.stages.equil_prep import run_equil_prep
from paddle.learn.data import read_prep_logs, make_windows, time_split, save_npz_bundle
from paddle.learn.model import TrainConfig, train_ensemble
from paddle.stages.equil_prod import run_equil_and_prod

def _resolve_outdir(base: str | None, default: str) -> Path:
    p = Path(base) if base else Path(default)
    p.mkdir(parents=True, exist_ok=True)
    return p

def cmd_cmd(ns):
    cfg = SimulationConfig.from_file(ns.config)
    if ns.out:
        cfg.outdir = ns.out
    cfg.validate()
    run_cmd(cfg)

def cmd_prep(ns):
    cfg = SimulationConfig.from_file(ns.config)
    if ns.out:
        cfg.outdir = ns.out
    cfg.validate()
    run_equil_prep(cfg)

def cmd_data(ns):
    prep_dir = Path(ns.prep)
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_prep_logs(prep_dir)
    feat_cols = [s.strip() for s in ns.features.split(",") if s.strip()]
    targ_cols = [s.strip() for s in ns.target.split(",") if s.strip()]
    X, y, stats = make_windows(df, feat_cols, targ_cols, ns.window, ns.stride, ns.horizon, ns.norm)
    idx = time_split(len(X), train=ns.train, val=ns.val)
    save_npz_bundle(out_dir, X, y, idx, stats)
    print(f"Dataset saved to: {out_dir}")

def cmd_train(ns):
    cfg = TrainConfig(
        epochs=ns.epochs, batch=ns.batch, ensemble=ns.ensemble,
        hidden=[int(x) for x in ns.hidden.split(",")] if ns.hidden else None,
        dropout=ns.dropout, patience=ns.patience, seed=ns.seed,
    )
    train_ensemble(ns.data, ns.splits, ns.out, cfg)

def cmd_equil_prod(ns):
    cfg = SimulationConfig.from_file(ns.config)
    if ns.out:
        cfg.outdir = ns.out
    cfg.validate()
    run_equil_and_prod(cfg)

def cmd_pipeline(ns):
    cfg = SimulationConfig.from_file(ns.config)
    out_root = _resolve_outdir(ns.out, "out_pipeline")
    cfg.outdir = str(out_root)
    cfg.validate()
    print("[1/5] CMD…"); run_cmd(cfg)
    print("[2/5] PREP…"); run_equil_prep(cfg)
    print("[3/5] DATA…")
    prep_dir = out_root / "prep"
    data_dir = out_root / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    df = read_prep_logs(prep_dir)
    feat_cols = [s.strip() for s in ns.features.split(",") if s.strip()]
    targ_cols = [s.strip() for s in ns.target.split(",") if s.strip()]
    X, y, stats = make_windows(df, feat_cols, targ_cols, ns.window, ns.stride, ns.horizon, ns.norm)
    idx = time_split(len(X), train=ns.train, val=ns.val)
    save_npz_bundle(data_dir, X, y, idx, stats)
    if not ns.skip_train:
        print("[4/5] TRAIN…")
        tcfg = TrainConfig(
            epochs=ns.epochs, batch=ns.batch, ensemble=ns.ensemble,
            hidden=[int(x) for x in ns.hidden.split(",")] if ns.hidden else None,
            dropout=ns.dropout, patience=ns.patience, seed=ns.seed,
        )
        model_dir = out_root / "models" / "run1"; model_dir.mkdir(parents=True, exist_ok=True)
        train_ensemble(data_dir / "windows.npz", data_dir / "splits.json", model_dir, tcfg)
    else:
        print("[4/5] TRAIN skipped.")
    print("[5/5] EQUIL+PROD…"); run_equil_and_prod(cfg)

def build_parser():
    ap = argparse.ArgumentParser(description="paddle CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cmd", help="Run CMD stage"); p.add_argument("--config", required=True); p.add_argument("--out", default=None); p.set_defaults(func=cmd_cmd)
    p = sub.add_parser("prep", help="Run equilibration-prep stage"); p.add_argument("--config", required=True); p.add_argument("--out", default=None); p.set_defaults(func=cmd_prep)
    p = sub.add_parser("data", help="Build dataset from prep logs")
    p.add_argument("--prep", required=True); p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=128); p.add_argument("--stride", type=int, default=4)
    p.add_argument("--features", default="Etot_kJ,Edih_kJ,T_K"); p.add_argument("--target", default="Etot_kJ")
    p.add_argument("--horizon", type=int, default=1); p.add_argument("--norm", choices=["zscore","minmax","none"], default="zscore")
    p.add_argument("--train", type=float, default=0.8); p.add_argument("--val", type=float, default=0.1)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="Train ensemble model")
    p.add_argument("--data", required=True); p.add_argument("--splits", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch", type=int, default=256); p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--hidden", default="256,256"); p.add_argument("--dropout", type=float, default=0.1); p.add_argument("--patience", type=int, default=8); p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("equil_prod", help="Run equilibration + production"); p.add_argument("--config", required=True); p.add_argument("--out", default=None); p.set_defaults(func=cmd_equil_prod)

    p = sub.add_parser("pipeline", help="Run full pipeline"); p.add_argument("--config", required=True); p.add_argument("--out", default=None)
    p.add_argument("--window", type=int, default=128); p.add_argument("--stride", type=int, default=4)
    p.add_argument("--features", default="Etot_kJ,Edih_kJ,T_K"); p.add_argument("--target", default="Etot_kJ")
    p.add_argument("--horizon", type=int, default=1); p.add_argument("--norm", choices=["zscore","minmax","none"], default="zscore")
    p.add_argument("--train", type=float, default=0.8); p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch", type=int, default=256); p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--hidden", default="256,256"); p.add_argument("--dropout", type=float, default=0.1); p.add_argument("--patience", type=int, default=8); p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--skip-train", action="store_true")
    p.set_defaults(func=cmd_pipeline)
    return ap

def main(argv=None) -> int:
    ap = build_parser()
    ns = ap.parse_args(argv)
    ns.func(ns)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
