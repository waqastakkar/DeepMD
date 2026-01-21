"""
io/restart.py — Read/write restart metadata for PADDLE dual-boost runs
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import math
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
    header: str = "#Parameters\tValues(kcal/mol)"
    comments_by_index: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
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


DEFAULT_K0_BOUNDS = (0.0, 1.0)
_TRAILING_COMMENT_INDEX = 11


def _ensure_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _normalize_comments(comments: Dict[int, Iterable[str]]) -> Dict[int, Tuple[str, ...]]:
    return {idx: tuple(lines) for idx, lines in comments.items()}

def _parse_value(label: str, value: str) -> float:
    try:
        return float(value.strip())
    except Exception as e:
        raise RestartFormatError(f"Invalid numeric value for {label!r}: {value!r}") from e

def read_restart(path: str | Path, k0_bounds: Tuple[float, float] = DEFAULT_K0_BOUNDS) -> RestartRecord:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Restart file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RestartFormatError("Empty restart file")
    if not lines[0].startswith("#"):
        raise RestartFormatError("Missing header line in restart file")
    header = lines[0]
    values: Dict[int, float] = {}
    comment_map: Dict[int, list[str]] = {}
    pending_comments: list[str] = []
    for line in lines[1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            pending_comments.append(line)
            continue
        try:
            left, right = line.split("\t", 1)
        except ValueError as e:
            raise RestartFormatError(f"Bad label format: {line!r}") from e
        if not left.startswith("(") or ")" not in left:
            raise RestartFormatError(f"Bad label format: {left!r}")
        idx_str = left.split(")", 1)[0][1:]
        idx = int(idx_str)
        if pending_comments:
            comment_map.setdefault(idx, []).extend(pending_comments)
            pending_comments = []
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
        header=header,
        comments_by_index=_normalize_comments(comment_map),
    )
    if pending_comments:
        rec.comments_by_index[_TRAILING_COMMENT_INDEX] = tuple(pending_comments)
    return normalize_restart_record(rec, k0_bounds=k0_bounds)

def write_restart(
    path: str | Path,
    rec: RestartRecord,
    k0_bounds: Tuple[float, float] = DEFAULT_K0_BOUNDS,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = normalize_restart_record(rec, k0_bounds=k0_bounds)
    steps, VminD, VmaxD, Dref, Dboost, k0D, VminP, VmaxP, Tref, Tboost, k0P = rec.to_kcal_tuple()
    lines = [rec.header]
    for idx in range(0, _TRAILING_COMMENT_INDEX):
        lines.extend(rec.comments_by_index.get(idx, ()))
        if idx == 0:
            lines.append(f"(0)Steps:\t{int(steps)}")
            continue
        if idx == 1:
            lines.append(f"(1)Boosted VminD:\t{VminD}")
            continue
        if idx == 2:
            lines.append(f"(2)Boosted VmaxD:\t{VmaxD}")
            continue
        if idx == 3:
            lines.append(f"(3)DihedralRefEnergy:\t{Dref}")
            continue
        if idx == 4:
            lines.append(f"(4)Final DihedralBoost:\t{Dboost}")
            continue
        if idx == 5:
            lines.append(f"(5)Final k0D:\t{k0D}")
            continue
        if idx == 6:
            lines.append(f"(6)Boosted VminP:\t{VminP}")
            continue
        if idx == 7:
            lines.append(f"(7)Boosted VmaxP:\t{VmaxP}")
            continue
        if idx == 8:
            lines.append(f"(8)TotalRefEnergy:\t{Tref}")
            continue
        if idx == 9:
            lines.append(f"(9)Final TotalBoost:\t{Tboost}")
            continue
        if idx == 10:
            lines.append(f"(10)Final k0P:\t{k0P}")
            continue
    lines.extend(rec.comments_by_index.get(_TRAILING_COMMENT_INDEX, ()))
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
    from paddle.core.params import BoostParams
    return BoostParams(
        VminD=rec.VminD_kJ,
        VmaxD=rec.VmaxD_kJ,
        VminP=rec.VminP_kJ,
        VmaxP=rec.VmaxP_kJ,
        k0D=rec.k0D,
        k0P=rec.k0P,
    )

def validate_against_state(rec: RestartRecord, expected_steps: Optional[int] = None) -> None:
    normalize_restart_record(rec)
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


def normalize_restart_record(
    rec: RestartRecord,
    k0_bounds: Tuple[float, float] = DEFAULT_K0_BOUNDS,
) -> RestartRecord:
    k0_min, k0_max = k0_bounds
    if k0_min > k0_max:
        raise ValueError(f"k0_min must be <= k0_max, got {k0_bounds}")
    _ensure_finite("steps", float(rec.steps))
    for label, value in [
        ("VminD_kJ", rec.VminD_kJ),
        ("VmaxD_kJ", rec.VmaxD_kJ),
        ("DihedralRef_kJ", rec.DihedralRef_kJ),
        ("DihedralBoost_kJ", rec.DihedralBoost_kJ),
        ("k0D", rec.k0D),
        ("VminP_kJ", rec.VminP_kJ),
        ("VmaxP_kJ", rec.VmaxP_kJ),
        ("TotalRef_kJ", rec.TotalRef_kJ),
        ("TotalBoost_kJ", rec.TotalBoost_kJ),
        ("k0P", rec.k0P),
    ]:
        _ensure_finite(label, value)
    if rec.VmaxD_kJ < rec.VminD_kJ:
        raise ValueError("VmaxD_kJ must be >= VminD_kJ")
    if rec.VmaxP_kJ < rec.VminP_kJ:
        raise ValueError("VmaxP_kJ must be >= VminP_kJ")
    clamped_k0D = _clamp(rec.k0D, k0_min, k0_max)
    clamped_k0P = _clamp(rec.k0P, k0_min, k0_max)
    if clamped_k0D != rec.k0D or clamped_k0P != rec.k0P:
        return replace(rec, k0D=clamped_k0D, k0P=clamped_k0P)
    return rec
