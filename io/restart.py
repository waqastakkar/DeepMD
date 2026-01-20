"""
io/restart.py — Read/write restart metadata for PADDLE dual-boost runs
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil

_KJ_PER_KCAL = 4.184

@dataclass
class RestartRecord:
    steps: int
    VminD_kJ: float
    VmaxD_kJ: float
    DihedralRef_kJ: float
    DihedralBoost_kJ: float
    k0D: float
    VminP_kJ: float
    VmaxP_kJ: float
    TotalRef_kJ: float
    TotalBoost_kJ: float
    k0P: float
    def to_kcal_tuple(self):
        return (
            self.steps,
            self.VminD_kJ / _KJ_PER_KCAL,
            self.VmaxD_kJ / _KJ_PER_KCAL,
            self.DihedralRef_kJ / _KJ_PER_KCAL,
            self.DihedralBoost_kJ / _KJ_PER_KCAL,
            self.k0D,
            self.VminP_kJ / _KJ_PER_KCAL,
            self.VmaxP_kJ / _KJ_PER_KCAL,
            self.TotalRef_kJ / _KJ_PER_KCAL,
            self.TotalBoost_kJ / _KJ_PER_KCAL,
            self.k0P,
        )

class RestartFormatError(RuntimeError):
    pass

def _parse_value(label: str, value: str) -> float:
    try:
        return float(value.strip())
    except Exception as e:
        raise RestartFormatError(f"Invalid numeric value for {label!r}: {value!r}") from e

def read_restart(path: str | Path) -> RestartRecord:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Restart file not found: {p}")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise RestartFormatError("Empty restart file")
    if not lines[0].startswith("#"):
        raise RestartFormatError("Missing header line in restart file")
    values: Dict[int, float] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        left, right = line.split("\t", 1)
        if not left.startswith("(") or ")" not in left:
            raise RestartFormatError(f"Bad label format: {left!r}")
        idx_str = left.split(")", 1)[0][1:]
        idx = int(idx_str)
        values[idx] = _parse_value(left, right)
    required = list(range(0, 11))
    missing = [k for k in required if k not in values]
    if missing:
        raise RestartFormatError(f"Missing indices in restart file: {missing}")
    steps = int(values[0])
    rec = RestartRecord(
        steps=steps,
        VminD_kJ=values[1] * _KJ_PER_KCAL,
        VmaxD_kJ=values[2] * _KJ_PER_KCAL,
        DihedralRef_kJ=values[3] * _KJ_PER_KCAL,
        DihedralBoost_kJ=values[4] * _KJ_PER_KCAL,
        k0D=float(values[5]),
        VminP_kJ=values[6] * _KJ_PER_KCAL,
        VmaxP_kJ=values[7] * _KJ_PER_KCAL,
        TotalRef_kJ=values[8] * _KJ_PER_KCAL,
        TotalBoost_kJ=values[9] * _KJ_PER_KCAL,
        k0P=float(values[10]),
    )
    return rec

def write_restart(path: str | Path, rec: RestartRecord) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    steps, VminD, VmaxD, Dref, Dboost, k0D, VminP, VmaxP, Tref, Tboost, k0P = rec.to_kcal_tuple()
    lines = [
        "#Parameters\tValues(kcal/mol)",
        f"(0)Steps:\t{int(steps)}",
        f"(1)Boosted VminD:\t{VminD}",
        f"(2)Boosted VmaxD:\t{VmaxD}",
        f"(3)DihedralRefEnergy:\t{Dref}",
        f"(4)Final DihedralBoost:\t{Dboost}",
        f"(5)Final k0D:\t{k0D}",
        f"(6)Boosted VminP:\t{VminP}",
        f"(7)Boosted VmaxP:\t{VmaxP}",
        f"(8)TotalRefEnergy:\t{Tref}",
        f"(9)Final TotalBoost:\t{Tboost}",
        f"(10)Final k0P:\t{k0P}",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p

def backup(path: str | Path, backup_dir: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / (p.stem + ".backup" + p.suffix)
    shutil.copy2(p, dst)
    return dst

def record_to_boost_params(rec: RestartRecord):
    from dbmdx.core.integrators import BoostParams
    return BoostParams(
        VminD=rec.VminD_kJ,
        VmaxD=rec.VmaxD_kJ,
        VminP=rec.VminP_kJ,
        VmaxP=rec.VmaxP_kJ,
        k0D=rec.k0D,
        k0P=rec.k0P,
    )

def validate_against_state(rec: RestartRecord, expected_steps: Optional[int] = None) -> None:
    if rec.steps < 0:
        raise RestartFormatError("steps must be non-negative")
    if rec.VmaxD_kJ < rec.VminD_kJ:
        raise RestartFormatError("VmaxD < VminD")
    if rec.VmaxP_kJ < rec.VminP_kJ:
        raise RestartFormatError("VmaxP < VminP")
    if not (0.0 < rec.k0D <= 1.0):
        raise RestartFormatError("k0D out of (0,1]")
    if not (0.0 < rec.k0P <= 1.0):
        raise RestartFormatError("k0P out of (0,1]")
    if expected_steps is not None and rec.steps != expected_steps:
        raise RestartFormatError(f"unexpected steps: {rec.steps} != {expected_steps}")
