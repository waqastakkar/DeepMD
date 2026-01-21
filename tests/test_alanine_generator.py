from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from benchmarks.alanine.generate_alanine import (
    EXPLICIT_TLEAP,
    IMPLICIT_TLEAP,
    _assert_tleap_available,
    generate_alanine,
)


def test_generate_alanine_writes_inputs_and_runs_tleap(tmp_path, monkeypatch):
    amberhome = tmp_path / "amber"
    amberhome.mkdir()
    monkeypatch.setenv("AMBERHOME", str(amberhome))

    calls = []

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, check=False):
        calls.append((cmd, Path(cwd)))
        if stdout:
            stdout.write("tleap ok\n")
        (Path(cwd) / "complex.parm7").write_text("parm7", encoding="utf-8")
        (Path(cwd) / "complex.rst7").write_text("rst7", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_root = tmp_path / "benchmarks"
    locations = generate_alanine(out_root)

    implicit_dir = locations["implicit"]
    explicit_dir = locations["explicit"]

    assert (implicit_dir / "tleap.in").read_text(encoding="utf-8") == IMPLICIT_TLEAP
    assert (explicit_dir / "tleap.in").read_text(encoding="utf-8") == EXPLICIT_TLEAP
    assert len(calls) == 2
    assert calls[0][0] == ["tleap", "-f", "tleap.in"]
    assert calls[1][0] == ["tleap", "-f", "tleap.in"]
    assert calls[0][1] == implicit_dir
    assert calls[1][1] == explicit_dir

    def read_config(path: Path) -> dict[str, str]:
        data: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
        return data

    implicit_cfg = read_config(implicit_dir / "config.yml")
    explicit_cfg = read_config(explicit_dir / "config.yml")

    assert implicit_cfg["parmFile"] == str((implicit_dir / "complex.parm7").resolve())
    assert implicit_cfg["crdFile"] == str((implicit_dir / "complex.rst7").resolve())
    assert explicit_cfg["parmFile"] == str((explicit_dir / "complex.parm7").resolve())
    assert explicit_cfg["crdFile"] == str((explicit_dir / "complex.rst7").resolve())


def test_assert_tleap_available_errors_when_missing(monkeypatch):
    monkeypatch.delenv("AMBERHOME", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="tleap"):
        _assert_tleap_available()
