"""
core/params.py — Shared parameter dataclasses
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoostParams:
    VminD: float
    VmaxD: float
    VminP: float
    VmaxP: float
    k0D: float
    k0P: float
    refED_factor: float = 0.05
    refEP_factor: float = 0.05

    def clamp(self) -> None:
        if self.VmaxD < self.VminD:
            self.VminD, self.VmaxD = self.VmaxD, self.VminD
        if self.VmaxP < self.VminP:
            self.VminP, self.VmaxP = self.VmaxP, self.VminP
        self.k0D = float(max(0.0, min(1.0, self.k0D)))
        self.k0P = float(max(0.0, min(1.0, self.k0P)))
