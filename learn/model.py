"""
learn/model.py — Deep ensemble with Gaussian heads for PADDLE
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

# Use a single, stable precision policy to avoid dtype mismatches
tf.keras.mixed_precision.set_global_policy("float32")


@register_keras_serializable(package="dbmdx")
def gaussian_nll(y_true, y_pred, eps: float = 1e-5):
    """Negative log-likelihood for diagonal Gaussian [mu, sigma]."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    D = tf.shape(y_true)[-1]
    mu = y_pred[:, :D]
    sigma = tf.maximum(y_pred[:, D:], eps)
    var = tf.square(sigma)
    # 0.5 * [ log(2π) + log(var) + (y-mu)^2 / var ]
    nll = 0.5 * (
        tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=tf.float32))
        + tf.math.log(var)
        + tf.square(y_true - mu) / var
    )
    return tf.reduce_mean(tf.reduce_sum(nll, axis=-1))


@register_keras_serializable(package="dbmdx")
def residual_kurtosis_excess(y_true, y_pred, eps: float = 1e-6):
    """|kurtosis(residuals) - 3| averaged across dims."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    D = tf.shape(y_true)[-1]
    mu = y_pred[:, :D]
    r = y_true - mu
    m2 = tf.reduce_mean(tf.square(r), axis=0) + eps
    m4 = tf.reduce_mean(tf.square(tf.square(r)), axis=0)
    kurt = m4 / (tf.square(m2) + eps)
    return tf.reduce_mean(tf.abs(kurt - 3.0))


def make_mlp(
    input_shape: Tuple[int, int],
    output_dim: int,
    hidden: List[int] | Tuple[int, ...] = (256, 256),
    activation: str = "gelu",
    dropout: float = 0.0,
) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="inp")
    x = layers.Flatten(name="flatten")(inp)
    for i, h in enumerate(hidden):
        x = layers.Dense(int(h), activation=activation, name=f"dense_{i}")(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(rate=float(dropout), name=f"dropout_{i}")(x)
    mu = layers.Dense(output_dim, name="mu")(x)
    # Use Dense with softplus activation instead of layers.Softplus
    sigma = layers.Dense(output_dim, activation="softplus", name="sigma")(x)
    out = layers.Concatenate(name="gaussian_head")([mu, sigma])
    model = models.Model(inp, out, name="mlp_gaussian")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=gaussian_nll,
        metrics=[residual_kurtosis_excess],
    )
    return model

def load_npz_bundle(path_npz: str | Path):
    data = np.load(path_npz)
    return data["X"], data["y"]


def make_datasets(npz_path: str | Path, splits_json: str | Path, batch: int = 256):
    X, y = load_npz_bundle(npz_path)
    with open(splits_json, "r", encoding="utf-8") as f:
        sp = json.load(f)

    def _sl(k):
        a, b = sp[k]
        return slice(a, b)

    Xtr, ytr = X[_sl("train")], y[_sl("train")]
    Xva, yva = X[_sl("val")], y[_sl("val")]
    Xte, yte = X[_sl("test")], y[_sl("test")]

    def _ds(Xa, ya):
        ds = (
            tf.data.Dataset.from_tensor_slices((Xa, ya))
            .shuffle(min(len(Xa), 65536))
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    # Return input shape as (window, features)
    return _ds(Xtr, ytr), _ds(Xva, yva), _ds(Xte, yte), (X.shape[1], X.shape[2])


@dataclass
class TrainConfig:
    epochs: int = 50
    batch: int = 256
    ensemble: int = 3
    hidden: List[int] | None = None
    dropout: float = 0.1
    patience: int = 8
    seed: int = 2025


def train_ensemble(npz: str | Path, splits: str | Path, out_dir: str | Path, cfg: TrainConfig) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_tr, ds_va, ds_te, in_shape = make_datasets(npz, splits, batch=cfg.batch)
    X, y = load_npz_bundle(npz)
    D = y.shape[1]

    results = {"members": [], "input_shape": in_shape, "output_dim": int(D)}

    with open(splits, "r", encoding="utf-8") as f:
        sp = json.load(f)
    a, b = sp["test"]

    for k in range(cfg.ensemble):
        tf.keras.utils.set_random_seed(cfg.seed + k)
        model = make_mlp(
            in_shape,
            D,
            hidden=(cfg.hidden or [256, 256]),
            dropout=cfg.dropout,
        )
        ckpt = out / f"member_{k}"
        ckpt.mkdir(parents=True, exist_ok=True)

        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                str(ckpt / "best.keras"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, cfg.patience // 2)
            ),
        ]

        hist = model.fit(ds_tr, validation_data=ds_va, epochs=cfg.epochs, callbacks=cb, verbose=2)
        model.save(str(ckpt / "final.keras"))
        (ckpt / "history.json").write_text(json.dumps(hist.history, indent=2), encoding="utf-8")
        results["members"].append({"path": str(ckpt)})

    mu_te, sig_te = ensemble_predict(out, X[a:b], batch=cfg.batch)
    y_te = y[a:b]
    metrics = compute_metrics(y_te, mu_te, sig_te)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    meta = {"train_config": asdict(cfg), **results, "metrics": metrics}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_members(model_dir: Path) -> list[tf.keras.Model]:
    """Load each ensemble member from best.keras (fallback to final.keras)."""
    members = []
    for sub in sorted(Path(model_dir).glob("member_*")):
        best = sub / "best.keras"
        final = sub / "final.keras"
        target = best if best.exists() else final
        if not target.exists():
            continue
        m = tf.keras.models.load_model(
            str(target),
            compile=False,
            custom_objects={
                "gaussian_nll": gaussian_nll,
                "residual_kurtosis_excess": residual_kurtosis_excess,
            },
            safe_mode=False,
        )
        members.append(m)
    if not members:
        raise RuntimeError(f"No members found in {model_dir}")
    return members


def ensemble_predict(model_dir: str | Path, X: np.ndarray, batch: int = 512):
    members = load_members(Path(model_dir))
    preds = []
    for m in members:
        y = m.predict(X, batch_size=batch, verbose=0)
        D = y.shape[1] // 2
        mu = y[:, :D]
        sigma = y[:, D:]
        preds.append((mu, sigma))

    mus = np.stack([p[0] for p in preds], axis=0)
    sigs = np.stack([p[1] for p in preds], axis=0)

    mu_ens = np.mean(mus, axis=0)
    ale = np.mean(np.square(sigs), axis=0)  # aleatoric variance
    epi = np.var(mus, axis=0)               # epistemic variance
    sigma_total = np.sqrt(ale + epi + 1e-9)
    return mu_ens.astype(np.float32), sigma_total.astype(np.float32)


def compute_metrics(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    resid = y_true - mu
    mse = float(np.mean(np.square(resid)))
    mae = float(np.mean(np.abs(resid)))
    within1 = float(np.mean(np.abs(resid) <= sigma))
    return {"mse": mse, "mae": mae, "within_1sigma": within1}


# ---- CLI (unchanged interface) ------------------------------------------------

@dataclass
class TrainConfigCLI:
    pass  # kept for clarity; argparse below builds TrainConfig directly


def _cmd_train(ns):
    cfg = TrainConfig(
        epochs=ns.epochs,
        batch=ns.batch,
        ensemble=ns.ensemble,
        hidden=[int(x) for x in ns.hidden.split(",")] if ns.hidden else None,
        dropout=ns.dropout,
        patience=ns.patience,
        seed=ns.seed,
    )
    train_ensemble(ns.data, ns.splits, ns.out, cfg)


def _cmd_eval(ns):
    X, y = load_npz_bundle(ns.data)
    with open(ns.splits, "r", encoding="utf-8") as f:
        sp = json.load(f)
    a, b = sp["test"]
    mu, sigma = ensemble_predict(ns.model, X[a:b])
    metrics = compute_metrics(y[a:b], mu, sigma)
    print(json.dumps(metrics, indent=2))
    (Path(ns.model) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _cmd_predict(ns):
    X, y = load_npz_bundle(ns.data)
    with open(ns.splits, "r", encoding="utf-8") as f:
        sp = json.load(f)
    a, b = sp["test"]
    mu, sigma = ensemble_predict(ns.model, X[a:b])
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mu=mu, sigma=sigma, y=y[a:b])
    print(f"Wrote predictions: {out}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DBMDX deep ensemble model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train ensemble")
    ap_train.add_argument("--data", required=True)
    ap_train.add_argument("--splits", required=True)
    ap_train.add_argument("--out", required=True)
    ap_train.add_argument("--epochs", type=int, default=50)
    ap_train.add_argument("--batch", type=int, default=256)
    ap_train.add_argument("--ensemble", type=int, default=3)
    ap_train.add_argument("--hidden", type=str, default="256,256")
    ap_train.add_argument("--dropout", type=float, default=0.1)
    ap_train.add_argument("--patience", type=int, default=8)
    ap_train.add_argument("--seed", type=int, default=2025)

    ap_eval = sub.add_parser("eval", help="Evaluate ensemble on test split")
    ap_eval.add_argument("--data", required=True)
    ap_eval.add_argument("--splits", required=True)
    ap_eval.add_argument("--model", required=True)

    ap_pred = sub.add_parser("predict", help="Export predictions for test split")
    ap_pred.add_argument("--data", required=True)
    ap_pred.add_argument("--splits", required=True)
    ap_pred.add_argument("--model", required=True)
    ap_pred.add_argument("--out", required=True)

    ns = ap.parse_args(argv)
    if ns.cmd == "train":
        _cmd_train(ns)
    elif ns.cmd == "eval":
        _cmd_eval(ns)
    elif ns.cmd == "predict":
        _cmd_predict(ns)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
