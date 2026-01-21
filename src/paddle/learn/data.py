"""
learn/data.py — Make training datasets from equil_prep CSV logs
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import argparse
import json
import random
import re
import numpy as np
import pandas as pd

def _list_prep_csvs(prep_dir: Path) -> List[Path]:
    files = sorted(prep_dir.glob("equilprep-cycle*.csv*"))
    if not files:
        raise FileNotFoundError(f"No prep CSV files found in: {prep_dir}")
    return files

def read_prep_logs(prep_dir: str | Path) -> pd.DataFrame:
    import re
    prep_dir = Path(prep_dir)
    rows = []
    for p in _list_prep_csvs(prep_dir):
        # robust cycle extraction for .csv or .csv.gz
        m = re.search(r"cycle(\d+)", p.name)
        if not m:
            raise ValueError(f"Cannot parse cycle index from filename: {p.name}")
        cyc = int(m.group(1))

        df = pd.read_csv(p, compression="infer")

        # Coerce expected columns to numeric
        for col, as_type in [("step", "int64"), ("Etot_kJ", "float64"),
                             ("Edih_kJ", "float64"), ("T_K", "float64")]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where step is missing; fill others safely
        df = df.dropna(subset=["step"]).copy()
        df["step"] = df["step"].astype(int)

        df["cycle"] = cyc
        rows.append(df)

    if not rows:
        raise RuntimeError("No rows read from prep logs.")

    df_all = pd.concat(rows, ignore_index=True)
    df_all.sort_values(["cycle", "step"], inplace=True)

    # Build global step
    base = 0
    gsteps = []
    last_cyc = None
    last_step = 0
    for cyc, step in zip(df_all["cycle"].to_numpy(), df_all["step"].to_numpy()):
        if last_cyc is None:
            last_cyc = cyc
        if cyc != last_cyc:
            base += last_step
            last_cyc = cyc
        gsteps.append(int(base + int(step)))
        last_step = int(step)
    df_all["gstep"] = gsteps

    return df_all.reset_index(drop=True)

def _compute_stats(arr: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
    mean = arr.mean(axis=axis)
    std = arr.std(axis=axis)
    std[std == 0] = 1.0
    minv = arr.min(axis=axis)
    maxv = arr.max(axis=axis)
    return {"mean": mean, "std": std, "min": minv, "max": maxv}

def _apply_norm(arr: np.ndarray, stats: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "none":
        return arr
    if mode == "zscore":
        return (arr - stats["mean"]) / stats["std"]
    if mode == "minmax":
        denom = (stats["max"] - stats["min"]).copy()
        denom[denom == 0] = 1.0
        return (arr - stats["min"]) / denom
    raise ValueError(f"Unknown norm mode: {mode}")

def make_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    window: int,
    stride: int = 1,
    horizon: int = 1,
    norm: str = "zscore",
):
    F = df[feature_cols].to_numpy(dtype=np.float32)
    Ysrc = df[target_cols].to_numpy(dtype=np.float32)
    stats_arr = _compute_stats(F, axis=0)
    stats = {
        "features": {k: stats_arr[k].astype(float).tolist() for k in stats_arr},
        "mode": {"norm": norm},
        "feature_cols": {"names": list(feature_cols)},
        "target_cols": {"names": list(target_cols)},
    }
    F_norm = _apply_norm(F, stats_arr, norm)

    Xs = []
    Ys = []
    Tlen = F_norm.shape[0]
    last_idx = Tlen - window - horizon + 1
    for start in range(0, last_idx, stride):
        end = start + window
        xt = F_norm[start:end, :]
        yt = Ysrc[end + horizon - 1, :]
        Xs.append(xt)
        Ys.append(yt)
    if not Xs:
        raise RuntimeError("Not enough points to form any window. Reduce window or horizon.")
    X = np.stack(Xs, axis=0)
    y = np.stack(Ys, axis=0)
    return X.astype(np.float32), y.astype(np.float32), stats

def time_split(N: int, train: float = 0.8, val: float = 0.1):
    train_n = int(N * train)
    val_n = int(N * val)
    test_n = N - train_n - val_n
    if min(train_n, val_n, test_n) <= 0:
        raise ValueError("Invalid split; adjust ratios.")
    idx = {"train": (0, train_n), "val": (train_n, train_n + val_n), "test": (train_n + val_n, N)}
    return idx

def split_arrays(X: np.ndarray, y: np.ndarray, idx):
    out = {}
    for k, (a, b) in idx.items():
        out[k] = (X[a:b], y[a:b])
    return out

def save_npz_bundle(out_dir: Path, X: np.ndarray, y: np.ndarray, idx, stats) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=X, y=y)
    (out_dir / "splits.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

def load_npz_bundle(path: str | Path):
    data = np.load(path)
    return data["X"], data["y"]

def project_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - mean) @ components.T

def load_latent_pca(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=float)
    components = np.asarray(payload["components"], dtype=float)
    return mean, components

def as_tf_dataset(X: np.ndarray, y: np.ndarray, batch: int = 256, shuffle: bool = True):
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        raise RuntimeError("TensorFlow is required for as_tf_dataset but is not installed") from e
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 65536), seed=1234, reshuffle_each_iteration=True)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DBMDX dataset builder")
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--features", type=str, default="Etot_kJ,Edih_kJ,T_K")
    ap.add_argument("--target", type=str, default="Etot_kJ")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    prep_dir = Path(args.prep)
    out_dir = Path(args.out)

    df = read_prep_logs(prep_dir)
    feat_cols = [s.strip() for s in args.features.split(",") if s.strip()]
    targ_cols = [s.strip() for s in args.target.split(",") if s.strip()]

    X, y, stats = make_windows(
        df=df,
        feature_cols=feat_cols,
        target_cols=targ_cols,
        window=args.window,
        stride=args.stride,
        horizon=args.horizon,
        norm=args.norm,
    )

    idx = time_split(len(X), train=args.train, val=args.val)
    save_npz_bundle(out_dir, X, y, idx, stats)

    print(f"Built dataset: X{X.shape}, y{y.shape}")
    print(f"Saved to: {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
