"""
GNN pipeline for GaMD-enhanced dynamics with residue-level graphs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


tf.keras.mixed_precision.set_global_policy("float32")


@dataclass
class GraphFrame:
    node_features: np.ndarray
    positions: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: Optional[np.ndarray] = None
    labels: Optional[Dict[str, np.ndarray]] = None


@dataclass
class GraphBatch:
    node_features: np.ndarray
    positions: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    frame_ids: np.ndarray
    global_features: Optional[np.ndarray] = None
    labels: Optional[Dict[str, np.ndarray]] = None


@dataclass
class GraphBuildConfig:
    contact_cutoff: float = 4.5
    include_self_edges: bool = False
    distance_features: bool = True
    direction_features: bool = True
    edge_feature_names: Tuple[str, ...] = (
        "contact",
        "hbond",
        "salt",
        "covariance",
    )


class GraphBuilder:
    def __init__(self, cfg: GraphBuildConfig):
        self.cfg = cfg

    def assemble_node_features(self, feature_dict: Mapping[str, np.ndarray]) -> np.ndarray:
        feats = [np.asarray(arr, dtype=np.float32) for arr in feature_dict.values()]
        if not feats:
            raise ValueError("No node features provided.")
        return np.concatenate(feats, axis=-1)

    def _contact_edges(self, positions: np.ndarray) -> np.ndarray:
        dist2 = self._pairwise_squared_distances(positions)
        cutoff2 = float(self.cfg.contact_cutoff) ** 2
        mask = dist2 <= cutoff2
        if not self.cfg.include_self_edges:
            np.fill_diagonal(mask, False)
        return np.argwhere(mask)

    def _pairwise_squared_distances(self, positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - positions[None, :, :]
        return np.sum(diff * diff, axis=-1)

    def _edge_feature_matrix(
        self,
        mask: Optional[np.ndarray],
        edge_shape: Tuple[int, int],
    ) -> np.ndarray:
        if mask is None:
            return np.zeros(edge_shape, dtype=np.float32)
        return np.asarray(mask, dtype=np.float32)

    def build_frame(
        self,
        node_features: np.ndarray,
        positions: np.ndarray,
        contact_mask: Optional[np.ndarray] = None,
        hbond_mask: Optional[np.ndarray] = None,
        salt_mask: Optional[np.ndarray] = None,
        covariance_mask: Optional[np.ndarray] = None,
        global_features: Optional[np.ndarray] = None,
        labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> GraphFrame:
        positions = np.asarray(positions, dtype=np.float32)
        node_features = np.asarray(node_features, dtype=np.float32)
        num_nodes = node_features.shape[0]

        if contact_mask is None:
            edge_index = self._contact_edges(positions)
            contact_values = np.ones((edge_index.shape[0], 1), dtype=np.float32)
        else:
            edge_index = np.argwhere(np.asarray(contact_mask, dtype=bool))
            contact_values = np.asarray(contact_mask, dtype=np.float32)[
                edge_index[:, 0], edge_index[:, 1]
            ][:, None]
        if not self.cfg.include_self_edges:
            mask = edge_index[:, 0] != edge_index[:, 1]
            edge_index = edge_index[mask]
            contact_values = contact_values[mask]

        contact = contact_values
        hbond = self._edge_feature_matrix(hbond_mask, (num_nodes, num_nodes))
        salt = self._edge_feature_matrix(salt_mask, (num_nodes, num_nodes))
        covar = self._edge_feature_matrix(covariance_mask, (num_nodes, num_nodes))

        features = []
        for name, mat in zip(
            self.cfg.edge_feature_names,
            (contact, hbond, salt, covar),
        ):
            if name:
                if mat.ndim == 2:
                    features.append(mat[edge_index[:, 0], edge_index[:, 1]][:, None])
                else:
                    features.append(mat)
        edge_features = np.concatenate(features, axis=-1).astype(np.float32)

        if self.cfg.distance_features or self.cfg.direction_features:
            rel = positions[edge_index[:, 1]] - positions[edge_index[:, 0]]
            distances = np.linalg.norm(rel, axis=-1, keepdims=True)
            if self.cfg.distance_features:
                edge_features = np.concatenate([edge_features, distances], axis=-1)
            if self.cfg.direction_features:
                unit = rel / np.clip(distances, 1e-6, None)
                edge_features = np.concatenate([edge_features, unit], axis=-1)

        return GraphFrame(
            node_features=node_features,
            positions=positions,
            edge_index=edge_index.astype(np.int32),
            edge_features=edge_features.astype(np.float32),
            global_features=None if global_features is None else np.asarray(global_features, dtype=np.float32),
            labels=labels,
        )


def batch_graphs(frames: Sequence[GraphFrame]) -> GraphBatch:
    node_features = []
    positions = []
    edge_index = []
    edge_features = []
    frame_ids = []
    global_features = []
    labels: Dict[str, List[np.ndarray]] = {}

    node_offset = 0
    for frame_id, frame in enumerate(frames):
        n_nodes = frame.node_features.shape[0]
        n_edges = frame.edge_index.shape[0]

        node_features.append(frame.node_features)
        positions.append(frame.positions)
        edge_features.append(frame.edge_features)
        edge_index.append(frame.edge_index + node_offset)
        frame_ids.append(np.full(n_nodes, frame_id, dtype=np.int32))

        if frame.global_features is not None:
            global_features.append(frame.global_features)
        if frame.labels:
            for key, value in frame.labels.items():
                labels.setdefault(key, []).append(np.asarray(value))

        node_offset += n_nodes
        _ = n_edges

    batch = GraphBatch(
        node_features=np.concatenate(node_features, axis=0),
        positions=np.concatenate(positions, axis=0),
        edge_index=np.concatenate(edge_index, axis=0),
        edge_features=np.concatenate(edge_features, axis=0),
        frame_ids=np.concatenate(frame_ids, axis=0),
        global_features=None if not global_features else np.stack(global_features, axis=0),
        labels=None if not labels else {k: np.stack(v, axis=0) for k, v in labels.items()},
    )
    return batch


def segment_softmax(logits: tf.Tensor, segment_ids: tf.Tensor, num_segments: tf.Tensor) -> tf.Tensor:
    max_per_segment = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
    max_per_edge = tf.gather(max_per_segment, segment_ids)
    stabilized = logits - max_per_edge
    exp = tf.exp(stabilized)
    sum_per_segment = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments)
    denom = tf.gather(sum_per_segment, segment_ids)
    return exp / (denom + 1e-8)


class GraphAttentionLayer(layers.Layer):
    def __init__(self, out_dim: int, heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.node_dense = layers.Dense(out_dim * heads, use_bias=False)
        self.edge_dense = layers.Dense(out_dim * heads, use_bias=False)
        self.attn_kernel = self.add_weight(
            name="attn_kernel",
            shape=(heads, out_dim * 3),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.out_proj = layers.Dense(out_dim)
        self.dropout_layer = layers.Dropout(dropout)
        self.last_attention = None

    def call(self, node_features: tf.Tensor, edge_index: tf.Tensor, edge_features: tf.Tensor, training=False):
        num_nodes = tf.shape(node_features)[0]
        edge_index = tf.cast(edge_index, tf.int32)
        num_edges = tf.shape(edge_index)[0]

        def _apply_attention():
            dst = edge_index[:, 1]
            order = tf.argsort(dst, stable=True)
            edge_index_sorted = tf.gather(edge_index, order)
            edge_features_sorted = tf.gather(edge_features, order)
            dst_sorted = tf.gather(dst, order)

            node_proj = self.node_dense(node_features)
            node_proj = tf.reshape(node_proj, (-1, self.heads, self.out_dim))
            edge_proj = self.edge_dense(edge_features_sorted)
            edge_proj = tf.reshape(edge_proj, (-1, self.heads, self.out_dim))

            src = tf.gather(node_proj, edge_index_sorted[:, 0])
            dst_feat = tf.gather(node_proj, edge_index_sorted[:, 1])
            attn_input = tf.concat([src, dst_feat, edge_proj], axis=-1)
            attn_logits = tf.einsum("ehd,hd->eh", attn_input, self.attn_kernel)

            attn_logits_t = tf.transpose(attn_logits, [1, 0])
            attn = tf.map_fn(
                lambda x: segment_softmax(x, dst_sorted, num_nodes),
                attn_logits_t,
                fn_output_signature=tf.float32,
            )
            attn = tf.transpose(attn, [1, 0])
            self.last_attention = attn

            messages = attn[:, :, None] * src
            messages = tf.reshape(messages, (-1, self.heads * self.out_dim))
            agg = tf.math.unsorted_segment_sum(messages, dst_sorted, num_nodes)

            out = self.out_proj(agg)
            out = tf.nn.gelu(out)
            out = self.dropout_layer(out, training=training)
            return out

        return tf.cond(num_edges > 0, _apply_attention, lambda: node_features)


class SE3MessagePassing(layers.Layer):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.edge_dense = layers.Dense(hidden_dim, activation="gelu")
        self.node_dense = layers.Dense(hidden_dim, activation="gelu")

    def call(self, node_features: tf.Tensor, positions: tf.Tensor, edge_index: tf.Tensor):
        edge_index = tf.cast(edge_index, tf.int32)
        rel = tf.gather(positions, edge_index[:, 1]) - tf.gather(positions, edge_index[:, 0])
        distances = tf.norm(rel, axis=-1, keepdims=True)
        rel_unit = rel / (distances + 1e-6)
        edge_repr = tf.concat([distances, rel_unit], axis=-1)
        edge_repr = self.edge_dense(edge_repr)

        src = tf.gather(node_features, edge_index[:, 0])
        msg = src * edge_repr
        agg = tf.math.unsorted_segment_sum(msg, edge_index[:, 1], tf.shape(node_features)[0])
        return self.node_dense(agg)


class GraphEncoder(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int = 128,
        gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre_dense = layers.Dense(hidden_dim, activation="gelu")
        self.se3 = SE3MessagePassing(hidden_dim)
        self.gat_stack = [
            GraphAttentionLayer(hidden_dim, heads=gat_heads, dropout=dropout)
            for _ in range(gat_layers)
        ]
        self.post_norm = layers.LayerNormalization()

    def call(
        self,
        node_features: tf.Tensor,
        positions: tf.Tensor,
        edge_index: tf.Tensor,
        edge_features: tf.Tensor,
        training=False,
    ) -> tf.Tensor:
        h = self.pre_dense(node_features)
        h = h + self.se3(h, positions, edge_index)
        for layer in self.gat_stack:
            h = h + layer(h, edge_index, edge_features, training=training)
        return self.post_norm(h)


class TemporalEncoder(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int = 128,
        conv_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_stack = [
            layers.Conv1D(hidden_dim, kernel_size=3, padding="same", activation="gelu")
            for _ in range(conv_layers)
        ]
        self.attn = layers.MultiHeadAttention(num_heads=attn_heads, key_dim=hidden_dim)
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization()

    def call(self, sequence: tf.Tensor, training=False) -> tf.Tensor:
        x = sequence
        for conv in self.conv_stack:
            x = conv(x)
        attn_out = self.attn(x, x)
        x = self.norm(x + attn_out)
        x = self.dropout(x, training=training)
        return x


class GaMDGNNModel(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 8,
        state_classes: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim=hidden_dim)
        self.pool = layers.GlobalAveragePooling1D()
        self.delta_v_head = layers.Dense(1, name="delta_v")
        self.state_head = layers.Dense(state_classes, activation="softmax", name="state")
        self.rmsd_head = layers.Dense(1, name="rmsd")
        self.rg_head = layers.Dense(1, name="rg")
        self.latent_head = layers.Dense(latent_dim, name="latent")

    def call(self, inputs: Mapping[str, tf.Tensor], training=False) -> Mapping[str, tf.Tensor]:
        node_features = inputs["node_features"]
        positions = inputs["positions"]
        edge_index = inputs["edge_index"]
        edge_features = inputs["edge_features"]
        frame_ids = tf.cast(inputs["frame_ids"], tf.int32)
        sequence_len = tf.cast(inputs["sequence_len"], tf.int32)

        node_embeddings = self.graph_encoder(
            node_features,
            positions,
            edge_index,
            edge_features,
            training=training,
        )

        frame_embeddings = tf.math.unsorted_segment_mean(
            node_embeddings, frame_ids, tf.reduce_max(frame_ids) + 1
        )
        total_frames = tf.shape(frame_embeddings)[0]
        batch_size = total_frames // sequence_len
        sequence = tf.reshape(frame_embeddings, (batch_size, sequence_len, -1))
        temporal = self.temporal_encoder(sequence, training=training)
        pooled = self.pool(temporal)

        return {
            "delta_v": self.delta_v_head(pooled),
            "state": self.state_head(pooled),
            "rmsd": self.rmsd_head(pooled),
            "rg": self.rg_head(pooled),
            "latent": self.latent_head(pooled),
        }


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 4
    sequence_len: int = 8
    learning_rate: float = 1e-3


class TrajectoryWindowDataset:
    def __init__(
        self,
        frames: Sequence[GraphFrame],
        labels: Mapping[str, np.ndarray],
        sequence_len: int,
        batch_size: int,
    ):
        self.frames = frames
        self.labels = labels
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.num_frames = len(frames)
        self.num_windows = self.num_frames - sequence_len + 1

    def _window_labels(self, start: int) -> Dict[str, np.ndarray]:
        end = start + self.sequence_len
        last_idx = end - 1
        return {k: v[last_idx] for k, v in self.labels.items()}

    def __iter__(self) -> Iterable[Tuple[GraphBatch, Dict[str, np.ndarray]]]:
        for start in range(0, self.num_windows, self.batch_size):
            batch_frames: List[GraphFrame] = []
            batch_labels: Dict[str, List[np.ndarray]] = {}
            count = 0
            for offset in range(self.batch_size):
                idx = start + offset
                if idx >= self.num_windows:
                    break
                window_frames = self.frames[idx : idx + self.sequence_len]
                batch_frames.extend(window_frames)
                labels = self._window_labels(idx)
                for key, value in labels.items():
                    batch_labels.setdefault(key, []).append(value)
                count += 1
            if count == 0:
                break
            batch = batch_graphs(batch_frames)
            yield batch, {k: np.stack(v, axis=0) for k, v in batch_labels.items()}


class SaliencyAnalyzer:
    def __init__(self, model: GaMDGNNModel):
        self.model = model

    def gradient_attribution(
        self,
        batch: GraphBatch,
        target: str = "delta_v",
    ) -> np.ndarray:
        inputs = _batch_to_tensor_inputs(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs["node_features"])
            outputs = self.model(inputs, training=False)[target]
            scalar = tf.reduce_sum(outputs)
        grads = tape.gradient(scalar, inputs["node_features"]).numpy()
        return grads

    def integrated_gradients(
        self,
        batch: GraphBatch,
        target: str = "delta_v",
        steps: int = 32,
    ) -> np.ndarray:
        inputs = _batch_to_tensor_inputs(batch)
        baseline = tf.zeros_like(inputs["node_features"])
        total_grads = tf.zeros_like(inputs["node_features"])
        for alpha in np.linspace(0.0, 1.0, steps):
            interp = baseline + alpha * (inputs["node_features"] - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                outputs = self.model({**inputs, "node_features": interp}, training=False)[target]
                scalar = tf.reduce_sum(outputs)
            grads = tape.gradient(scalar, interp)
            total_grads += grads
        return (inputs["node_features"] - baseline) * total_grads / float(steps)

    def attention_rollout(self) -> List[np.ndarray]:
        rollout = []
        for layer in self.model.graph_encoder.gat_stack:
            if layer.last_attention is not None:
                rollout.append(layer.last_attention.numpy())
        return rollout

    def graphcam(self, batch: GraphBatch) -> np.ndarray:
        inputs = _batch_to_tensor_inputs(batch)
        with tf.GradientTape() as tape:
            node_embeddings = self.model.graph_encoder(
                inputs["node_features"],
                inputs["positions"],
                inputs["edge_index"],
                inputs["edge_features"],
                training=False,
            )
            tape.watch(node_embeddings)
            frame_embeddings = tf.math.unsorted_segment_mean(
                node_embeddings, inputs["frame_ids"], tf.reduce_max(inputs["frame_ids"]) + 1
            )
            total_frames = tf.shape(frame_embeddings)[0]
            sequence_len = tf.cast(inputs["sequence_len"], tf.int32)
            batch_size = total_frames // sequence_len
            sequence = tf.reshape(frame_embeddings, (batch_size, sequence_len, -1))
            temporal = self.model.temporal_encoder(sequence, training=False)
            pooled = self.model.pool(temporal)
            outputs = self.model.delta_v_head(pooled)
            scalar = tf.reduce_sum(outputs)
        grads = tape.gradient(scalar, node_embeddings)
        weights = tf.reduce_mean(grads, axis=0, keepdims=True)
        cam = tf.reduce_sum(node_embeddings * weights, axis=-1)
        return cam.numpy()

    def energy_flow_attribution(self, batch: GraphBatch) -> np.ndarray:
        if batch.edge_features.shape[1] == 0:
            raise ValueError("Edge features required for energy flow attribution.")
        edge_weights = batch.edge_features[:, 0]
        src = batch.edge_index[:, 0]
        dst = batch.edge_index[:, 1]
        num_nodes = batch.node_features.shape[0]
        flow = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(flow, dst, edge_weights)
        np.add.at(flow, src, edge_weights)
        return flow


def _batch_to_tensor_inputs(batch: GraphBatch) -> Dict[str, tf.Tensor]:
    return {
        "node_features": tf.convert_to_tensor(batch.node_features, dtype=tf.float32),
        "positions": tf.convert_to_tensor(batch.positions, dtype=tf.float32),
        "edge_index": tf.convert_to_tensor(batch.edge_index, dtype=tf.int32),
        "edge_features": tf.convert_to_tensor(batch.edge_features, dtype=tf.float32),
        "frame_ids": tf.convert_to_tensor(batch.frame_ids, dtype=tf.int32),
        "sequence_len": tf.convert_to_tensor(1, dtype=tf.int32),
    }


def train_gamd_gnn(
    frames: Sequence[GraphFrame],
    labels: Mapping[str, np.ndarray],
    cfg: TrainingConfig,
    out_dir: str | Path,
    hidden_dim: int = 128,
    latent_dim: int = 8,
    state_classes: int = 4,
) -> GaMDGNNModel:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrajectoryWindowDataset(
        frames,
        labels,
        sequence_len=cfg.sequence_len,
        batch_size=cfg.batch_size,
    )

    model = GaMDGNNModel(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        state_classes=state_classes,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        losses = []
        for batch, batch_labels in dataset:
            inputs = _batch_to_tensor_inputs(batch)
            inputs["sequence_len"] = tf.convert_to_tensor(cfg.sequence_len, dtype=tf.int32)
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                loss = _compute_loss(outputs, batch_labels)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(loss.numpy())
        print(f"Epoch {epoch + 1}/{cfg.epochs} loss={float(np.mean(losses)):.4f}")

    model.save(out_dir / "gamd_gnn_model.keras")
    return model


def _compute_loss(outputs: Mapping[str, tf.Tensor], labels: Mapping[str, np.ndarray]) -> tf.Tensor:
    loss = 0.0
    if "delta_v" in labels:
        loss += tf.reduce_mean(tf.square(outputs["delta_v"] - labels["delta_v"]))
    if "rmsd" in labels:
        loss += tf.reduce_mean(tf.square(outputs["rmsd"] - labels["rmsd"]))
    if "rg" in labels:
        loss += tf.reduce_mean(tf.square(outputs["rg"] - labels["rg"]))
    if "state" in labels:
        loss += tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels["state"], outputs["state"])
        )
    if "latent" in labels:
        loss += tf.reduce_mean(tf.square(outputs["latent"] - labels["latent"]))
    return loss


def save_importance_maps(
    residue_ids: Sequence[int],
    importance: np.ndarray,
    out_path: str | Path,
    label: str,
) -> None:
    df = pd.DataFrame({"residue": residue_ids, "importance": importance})
    df.to_csv(out_path, index=False)
    meta = {"label": label, "num_residues": len(residue_ids)}
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_network_edges(
    edge_index: np.ndarray,
    weights: np.ndarray,
    out_path: str | Path,
) -> None:
    df = pd.DataFrame(
        {
            "src": edge_index[:, 0],
            "dst": edge_index[:, 1],
            "weight": weights,
        }
    )
    df.to_csv(out_path, index=False)


def compute_umap_embedding(latent: np.ndarray, out_path: str | Path) -> None:
    payload = {
        "latent": np.asarray(latent, dtype=float).tolist(),
        "note": "Use umap-learn offline for 2D/3D embedding.",
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
