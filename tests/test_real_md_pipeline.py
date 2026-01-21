import os
from pathlib import Path
import subprocess
import sys

import pytest

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency guard
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "cli.py"), *args]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _write_short_config(config_path: Path, outdir: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for the real MD pipeline test.")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data.update(
        {
            "outdir": str(outdir),
            "platform": "CPU",
            "require_gpu": False,
            "precision": "single",
            "cuda_precision": "single",
            "ntcmd": 5000,
            "cmdRestartFreq": 500,
            "ncycebprepstart": 0,
            "ncycebprepend": 1,
            "ntebpreppercyc": 5000,
            "ebprepRestartFreq": 500,
            "ncycebstart": 0,
            "ncycebend": 1,
            "ntebpercyc": 5000,
            "ebRestartFreq": 500,
            "ncycprodstart": 0,
            "ncycprodend": 1,
            "ntprodpercyc": 5000,
            "prodRestartFreq": 500,
        }
    )
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_real_md_pipeline(tmp_path: Path) -> None:
    if os.environ.get("RUN_REAL_MD_TESTS") != "1":
        pytest.skip("Set RUN_REAL_MD_TESTS=1 to enable the real MD pipeline test.")
    try:
        import openmm  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency import guard
        pytest.skip(f"OpenMM is required for real MD tests: {exc}")

    bench_root = tmp_path / "alanine_bench"
    _run_cli(["bench_alanine", "--out", str(bench_root)])

    implicit_dir = bench_root / "implicit"
    config_path = implicit_dir / "config.yml"
    outdir = tmp_path / "alanine_out"
    _write_short_config(config_path, outdir)

    _run_cli(["cmd", "--config", str(config_path)])
    _run_cli(["prep", "--config", str(config_path)])
    data_dir = outdir / "data"
    _run_cli(["data", "--prep", str(outdir / "prep"), "--out", str(data_dir)])
    model_dir = outdir / "models" / "run1"
    _run_cli(
        [
            "train",
            "--data",
            str(data_dir / "windows.npz"),
            "--splits",
            str(data_dir / "splits.json"),
            "--out",
            str(model_dir),
            "--epochs",
            "1",
            "--ensemble",
            "1",
        ]
    )
    _run_cli(["equil_prod", "--config", str(config_path)])

    assert outdir.exists()
    assert (outdir / "prep").exists()
    assert data_dir.exists()
    assert model_dir.exists()
    assert (outdir / "equil").exists()
    assert (outdir / "prod").exists()

    md_log = outdir / "md.log"
    assert md_log.exists()
    log_text = md_log.read_text(encoding="utf-8")
    assert "Step" in log_text
    assert "Temperature" in log_text

    assert (data_dir / "windows.npz").exists()
    assert list(model_dir.rglob("*.keras"))

    assert (outdir / "gamd-restart.dat").exists()
    assert (outdir / "cmd.rst").exists()
