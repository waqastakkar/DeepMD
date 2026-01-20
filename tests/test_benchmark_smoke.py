import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_alanine_benchmark_smoke(tmp_path):
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy is required for benchmark smoke test")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "alanine" / "run_benchmark.py"
    outdir = tmp_path / "alanine_out"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            "200",
            "--bins",
            "20",
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=repo_root,
    )

    pmf_path = outdir / "pmf.json"
    metrics_path = outdir / "metrics.json"
    runtime_path = outdir / "runtime.json"

    assert pmf_path.exists()
    assert metrics_path.exists()
    assert runtime_path.exists()

    json.loads(pmf_path.read_text(encoding="utf-8"))
    json.loads(metrics_path.read_text(encoding="utf-8"))
    json.loads(runtime_path.read_text(encoding="utf-8"))
