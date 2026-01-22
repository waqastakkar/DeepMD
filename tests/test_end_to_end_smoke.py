from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from paddle.config import SimulationConfig
from paddle.core.engine import EngineOptions, create_simulation
from paddle.core.integrators import make_conventional
from paddle.learn.data import make_windows, read_prep_logs, save_npz_bundle, time_split
from paddle.learn.gnn_pipeline import (
    GraphBuildConfig,
    GraphBuilder,
    SaliencyAnalyzer,
    TrajectoryWindowDataset,
    _batch_to_tensor_inputs,
    _compute_loss,
    GaMDGNNModel,
)
from paddle.stages.cmd import run_cmd
from paddle.stages.equil_prep import run_equil_prep
from paddle.stages.equil_prod import run_equil_and_prod


def _cuda_available(openmm) -> bool:
    return any(
        openmm.Platform.getPlatform(i).getName() == "CUDA"
        for i in range(openmm.Platform.getNumPlatforms())
    )


def _require_dependencies() -> tuple[object, object]:
    try:
        import openmm  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency import guard
        pytest.skip(f"OpenMM is required for the pipeline smoke test: {exc}")

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency import guard
        pytest.skip(f"TensorFlow is required for the GNN smoke test: {exc}")

    return openmm, tf


def _generate_alanine(tmp_path: Path) -> dict[str, Path]:
    try:
        from benchmarks.alanine.generate_alanine import generate_alanine
    except Exception as exc:  # pragma: no cover - optional dependency guard
        pytest.skip(f"Alanine generator import failed: {exc}")
    if not shutil.which("tleap"):
        pytest.skip("AmberTools tleap not available; skipping alanine generation.")
    try:
        return generate_alanine(tmp_path)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_end_to_end_smoke(tmp_path: Path) -> None:
    openmm, tf = _require_dependencies()
    tf.keras.utils.set_random_seed(2025)
    np.random.seed(2025)

    locations = _generate_alanine(tmp_path / "alanine")
    implicit_dir = locations["implicit"]
    explicit_dir = locations["explicit"]

    platform_name = "CUDA" if _cuda_available(openmm) else "CPU"
    outdir = tmp_path / "pipeline_out"

    cfg = SimulationConfig(
        parmFile=str(implicit_dir / "complex.parm7"),
        crdFile=str(implicit_dir / "complex.rst7"),
        simType="protein.implicit",
        outdir=str(outdir),
        platform=platform_name,
        require_gpu=False,
        precision="single",
        cuda_precision="single",
        dt=0.002,
        ntcmd=2000,
        cmdRestartFreq=200,
        ncycebprepstart=0,
        ncycebprepend=1,
        ntebpreppercyc=2000,
        ebprepRestartFreq=200,
        ncycebstart=0,
        ncycebend=1,
        ntebpercyc=2000,
        ebRestartFreq=200,
        ncycprodstart=0,
        ncycprodend=1,
        ntprodpercyc=2000,
        prodRestartFreq=200,
        do_minimize=False,
        do_heating=False,
        do_density_equil=False,
        heat_ns=0.0,
        ntheat=1,
        density_ns=0.0,
        ntdensity=1,
        pressure_atm=1.0,
        barostat_interval=25,
        compress_logs=False,
    )
    cfg.validate()

    run_cmd(cfg)
    run_equil_prep(cfg)
    run_equil_and_prod(cfg)

    md_log = outdir / "md.log"
    assert md_log.exists()

    prep_csv = outdir / "prep" / "equilprep-cycle00.csv"
    assert prep_csv.exists()

    equil_diag = outdir / "equil" / "gamd-diagnostics-cycle00.csv"
    prod_diag = outdir / "prod" / "gamd-diagnostics-cycle00.csv"
    assert equil_diag.exists()
    assert prod_diag.exists()

    import pandas as pd

    equil_df = pd.read_csv(equil_diag)
    for col in ("E_dihedral_kJ", "DeltaV_kJ"):
        assert col in equil_df.columns
    assert (equil_df["E_dihedral_kJ"].abs() > 1e-6).any()
    assert (equil_df["DeltaV_kJ"].abs() > 1e-6).any()

    data_dir = outdir / "data"
    df = read_prep_logs(outdir / "prep")
    X, y, stats = make_windows(
        df,
        [
            "E_potential_kJ",
            "E_bond_kJ",
            "E_angle_kJ",
            "E_dihedral_kJ",
            "E_nonbonded_kJ",
            "T_K",
        ],
        ["Etot_kJ"],
        window=8,
        stride=2,
        horizon=1,
        norm="zscore",
    )
    idx = time_split(len(X), train=0.8, val=0.1)
    save_npz_bundle(data_dir, X, y, idx, stats)
    assert (data_dir / "windows.npz").exists()

    dcd_equil = outdir / "equil" / "equil-cycle00.dcd"
    dcd_prod = outdir / "prod" / "prod-cycle00.dcd"
    assert dcd_equil.exists()
    assert dcd_prod.exists()

    opts = EngineOptions(
        sim_type="protein.explicit",
        platform_name=platform_name,
        precision="single",
        cuda_precision="single",
        add_barostat=True,
        barostat_pressure_atm=cfg.pressure_atm,
        barostat_interval=cfg.barostat_interval,
        barostat_temperature_kelvin=cfg.temperature,
    )
    integ = make_conventional(dt_ps=0.002, temperature_K=cfg.temperature, collision_rate_ps=1.0)
    sim = create_simulation(
        str(explicit_dir / "complex.parm7"),
        str(explicit_dir / "complex.rst7"),
        integ,
        opts,
    )
    has_barostat = any(
        isinstance(force, openmm.MonteCarloBarostat) for force in sim.system.getForces()
    )
    assert has_barostat

    rng = np.random.default_rng(2025)
    num_frames = 6
    num_nodes = 5
    node_features = rng.normal(size=(num_frames, num_nodes, 4)).astype(np.float32)
    positions = rng.normal(size=(num_frames, num_nodes, 3)).astype(np.float32)
    labels = {"delta_v": np.linspace(0.0, 1.0, num_frames, dtype=np.float32)[:, None]}

    builder = GraphBuilder(GraphBuildConfig())
    frames = [
        builder.build_frame(node_features[i], positions[i])
        for i in range(num_frames)
    ]
    dataset = TrajectoryWindowDataset(frames, labels, sequence_len=2, batch_size=1)
    batch, batch_labels = next(iter(dataset))

    model = GaMDGNNModel(hidden_dim=16, latent_dim=4, state_classes=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    inputs = _batch_to_tensor_inputs(batch)
    inputs["sequence_len"] = tf.convert_to_tensor(2, dtype=tf.int32)
    outputs = model(inputs, training=True)
    loss0 = float(_compute_loss(outputs, batch_labels).numpy())

    for _ in range(5):
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = _compute_loss(outputs, batch_labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    outputs = model(inputs, training=False)
    loss1 = float(_compute_loss(outputs, batch_labels).numpy())
    assert loss1 <= loss0 or math.isclose(loss0, loss1, rel_tol=1e-2)

    analyzer = SaliencyAnalyzer(model)
    grads = analyzer.gradient_attribution(batch)
    assert np.isfinite(grads).all()
