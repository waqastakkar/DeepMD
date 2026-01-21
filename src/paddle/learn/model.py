"""
learn/model.py — Deep ensemble with Gaussian heads for PADDLE
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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

    latent_model_summary = None
    try:
        train_slice = slice(*sp["train"])
        X_train = X[train_slice]
        if X_train.ndim == 3:
            X_train = X_train.reshape(-1, X_train.shape[-1])
        if X_train.ndim != 2:
            raise ValueError(f"Unexpected X_train shape for PCA: {X_train.shape}")
        if X_train.shape[0] < 2:
            raise ValueError("Not enough samples to fit PCA.")
        mean = np.mean(X_train, axis=0)
        X_centered = X_train - mean
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k_latent = int(getattr(cfg, "latent_pca_k", 3))
        k_latent = max(1, min(k_latent, Vt.shape[0]))
        components = Vt[:k_latent, :]
        if X_train.shape[0] > 1:
            var = (S ** 2) / (X_train.shape[0] - 1)
            total_var = float(np.sum(var))
            if total_var > 0:
                explained_ratio = (var[:k_latent] / total_var).tolist()
            else:
                explained_ratio = [0.0] * k_latent
        else:
            explained_ratio = [0.0] * k_latent
        pca_payload = {
            "mean": mean.astype(float).tolist(),
            "components": components.astype(float).tolist(),
            "explained_variance_ratio": explained_ratio,
        }
        (out / "latent_pca.json").write_text(json.dumps(pca_payload, indent=2), encoding="utf-8")
        latent_model_summary = {"type": "pca", "k": k_latent, "path": "latent_pca.json"}
    except Exception as exc:
        print(f"Warning: latent PCA fit failed; skipping. Error: {exc}")

    mu_te, sig_te = ensemble_predict(out, X[a:b], batch=cfg.batch)
    y_te = y[a:b]
    metrics = compute_metrics(y_te, mu_te, sig_te)
    try:
        a_val, b_val = sp["val"]
        mu_val, sig_val = ensemble_predict(out, X[a_val:b_val], batch=cfg.batch)
        alpha = 0.1
        qhat = conformal_qhat(y[a_val:b_val], mu_val, sig_val, alpha=alpha)
        diag = conformal_diagnostics(y_te, mu_te, sig_te, qhat)
        metrics["conformal_alpha"] = float(alpha)
        metrics["conformal_qhat"] = float(qhat)
        metrics["conformal_coverage_test"] = float(diag["conformal_coverage"])
        metrics["conformal_mean_halfwidth_test"] = float(diag["conformal_mean_halfwidth"])
        metrics["mean_sigma_uncalibrated"] = float(np.mean(sig_te))
        metrics["uncertainty"] = float(diag["conformal_mean_halfwidth"])
    except Exception as exc:
        print(
            "Warning: conformal calibration failed; using uncalibrated uncertainty. "
            f"Error: {exc}"
        )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_model_summary(out, metrics, latent_model=latent_model_summary)

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
        n = X.shape[0]
        pad = (-n) % batch
        if pad:
            # Fixed TensorFlow retracing by enforcing stable prediction signature.
            pad_shape = (pad, *X.shape[1:])
            X_batch = np.concatenate([X, np.zeros(pad_shape, dtype=X.dtype)], axis=0)
        else:
            X_batch = X
        y = m.predict(X_batch, batch_size=batch, verbose=0)[:n]
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
    mean_sigma = float(np.mean(sigma))
    return {
        "mse": mse,
        "mae": mae,
        "within_1sigma": within1,
        "uncertainty": mean_sigma,
    }


def conformal_qhat(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.1
) -> float:
    """
    Compute conformal scaling factor qhat for predictive intervals mu ± qhat*sigma.
    Uses validation residual scores s_i = max_j |y_i,j - mu_i,j| / sigma_i,j (joint across target dims).
    qhat is the (1-alpha) conformal quantile using the standard finite-sample correction:
        k = ceil((n + 1) * (1 - alpha))
        qhat = sorted(scores)[k-1]
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    eps = 1e-6
    sigma_safe = np.maximum(sigma, eps)
    resid = np.abs(y_true - mu) / sigma_safe
    scores = np.max(resid, axis=1)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("No samples available for conformal calibration.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.sort(scores)[k - 1])
    return qhat


def conformal_diagnostics(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, qhat: float
) -> Dict[str, float]:
    """
    Return coverage and mean interval half-width under mu ± qhat*sigma.
    Coverage is joint across dims: all(|resid| <= qhat*sigma) per sample.
    Mean half-width is mean(qhat*sigma).
    """
    eps = 1e-6
    sigma_safe = np.maximum(sigma, eps)
    resid = np.abs(y_true - mu)
    covered = np.all(resid <= qhat * sigma_safe, axis=1)
    coverage = float(np.mean(covered))
    mean_halfwidth = float(np.mean(qhat * sigma_safe))
    return {
        "conformal_coverage": coverage,
        "conformal_mean_halfwidth": mean_halfwidth,
    }


def export_model_summary(
    outdir: str | Path,
    metrics_dict: Mapping[str, object],
    latent_model: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {}
    uncertainty = metrics_dict.get("uncertainty")
    if uncertainty is None:
        uncertainty = metrics_dict.get("mean_sigma")
    if uncertainty is not None:
        summary["uncertainty"] = float(uncertainty)

    calibration = metrics_dict.get("calibration_score")
    if calibration is None:
        calibration = metrics_dict.get("within_1sigma")
    if calibration is not None:
        summary["calibration_score"] = float(calibration)

    fit_stats: Dict[str, float] = {}
    for key in ("mse", "mae", "rmse"):
        if key in metrics_dict and metrics_dict[key] is not None:
            fit_stats[key] = float(metrics_dict[key])
    if fit_stats:
        summary["fit_stats"] = fit_stats

    recommended_step = metrics_dict.get("recommended_step_size")
    if recommended_step is not None:
        summary["recommended_step_size"] = float(recommended_step)

    conformal_fields = (
        "conformal_alpha",
        "conformal_qhat",
        "conformal_coverage_test",
        "conformal_mean_halfwidth_test",
        "mean_sigma_uncalibrated",
    )
    for key in conformal_fields:
        if key in metrics_dict and metrics_dict[key] is not None:
            summary[key] = float(metrics_dict[key])

    if latent_model is not None:
        summary["latent_model"] = dict(latent_model)

    (out / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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
