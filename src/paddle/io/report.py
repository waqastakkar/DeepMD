"""
io/report.py — Logging utilities for PADDLE
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping
import csv, gzip, io, json, os, shutil, tempfile
from datetime import datetime

ISO_FMT = "%Y-%m-%dT%H-%M-%S"

def timestamp() -> str:
    return datetime.utcnow().strftime(ISO_FMT)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class CSVLogger:
    path: Path
    fieldnames: Iterable[str]
    compress: bool = True

    def __post_init__(self) -> None:
        ensure_dir(self.path.parent)
        self._fieldnames = list(self.fieldnames)
        self._exists = self.path.exists() and self.path.stat().st_size > 0

    def _open(self):
        if self.compress and self.path.suffix != ".gz":
            self.path = Path(str(self.path) + ".gz")
        if self.path.suffix == ".gz":
            return gzip.open(self.path, mode="at", encoding="utf-8")
        return open(self.path, mode="a", encoding="utf-8", newline="")

    def writerow(self, row: Mapping[str, object]) -> None:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix="csvlog_", suffix=".tmp", dir=str(self.path.parent))
        os.close(tmp_fd)
        try:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=self._fieldnames, dialect="excel")
            if not self._exists and (not self.path.exists() or self.path.stat().st_size == 0):
                writer.writeheader()
                self._exists = True
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})
            content = buf.getvalue()
            with self._open() as fh:
                fh.write(content)
        finally:
            try: os.remove(tmp_name)
            except OSError: pass

def write_json(data: Mapping[str, object], path: Path) -> Path:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path

def append_metrics(metrics: Mapping[str, object], outdir: Path, name: str = "metrics.jsonl") -> Path:
    ensure_dir(outdir)
    path = outdir / name
    line = json.dumps({"t": timestamp(), **metrics}, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path

def write_run_manifest(outdir: Path, info: Mapping[str, object]) -> Path:
    return write_json({"created": timestamp(), **info}, outdir / "run-manifest.json")
