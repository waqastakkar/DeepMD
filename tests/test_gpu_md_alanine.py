import json
import math
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paddle.bench.mdtest import run_alanine_mdtest


def _require_cuda(openmm) -> None:
    names = {
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    }
    if "CUDA" not in names:
        pytest.fail(
            "CUDA platform not available; GPU tests require a CUDA-enabled OpenMM build."
        )


def _read_last_log_values(log_path: Path) -> dict:
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines()]
    data_lines = [line for line in lines if line and not line.startswith("#")]
    if not data_lines:
        raise AssertionError("No data rows found in md.log")
    header = [col.strip() for col in lines[0].lstrip("#").split(",")]
    values = [val.strip() for val in data_lines[-1].split(",")]
    if len(header) != len(values):
        raise AssertionError("Mismatch between header and data columns in md.log")
    return {key: float(value) for key, value in zip(header, values)}


@pytest.mark.gpu
def test_openmm_cuda_alanine_mdtest(tmp_path: Path) -> None:
    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable CUDA tests.")
    import openmm
    import yaml  # type: ignore

    _require_cuda(openmm)

    base_config = ROOT / "benchmarks" / "alanine_mdtest" / "config.yml"
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["steps_total"] = 25000
    cfg["report_interval"] = 5000
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["write_trajectory"] = False

    cfg_path = tmp_path / "mdtest_config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    outdir = run_alanine_mdtest(cfg_path)

    state_path = outdir / "state.xml"
    metadata_path = outdir / "metadata.json"
    log_path = outdir / "md.log"

    assert state_path.exists()
    assert metadata_path.exists()
    assert log_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["platform"] == "CUDA"

    last_values = _read_last_log_values(log_path)
    for key in ("Potential Energy (kJ/mol)", "Kinetic Energy (kJ/mol)", "Temperature (K)"):
        assert math.isfinite(last_values[key])
