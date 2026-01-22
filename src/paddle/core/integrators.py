"""
core/integrators.py — Custom OpenMM integrators for PADDLE
"""
from __future__ import annotations

from openmm import CustomIntegrator, LangevinMiddleIntegrator, unit

from paddle.core.params import BoostParams

_EPS = 1e-12

class ConventionalMDIntegrator(CustomIntegrator):
    def __init__(self, dt_ps: float = 0.002, temperature_K: float = 300.0, collision_rate_ps: float = 1.0):
        super().__init__(dt_ps * unit.picoseconds)
        self.addGlobalVariable("collision_rate", collision_rate_ps / unit.picosecond)
        self.addGlobalVariable("thermal_energy", (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature_K * unit.kelvin))
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()
        self.addComputeGlobal("vscale", "exp(-dt*collision_rate)")
        self.addComputeGlobal("fscale", "(1-vscale)/collision_rate")
        self.addComputeGlobal("noisescale", "sqrt(thermal_energy*(1-vscale*vscale))")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("x", "x + dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")

class DualDeepLearningGaMDEquilibration(CustomIntegrator):
    def __init__(self, dt_ps: float = 0.002, temperature_K: float = 300.0, params: BoostParams | None = None):
        super().__init__(dt_ps * unit.picoseconds)
        if params is None:
            params = BoostParams(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        params.clamp()
        self.addGlobalVariable("collision_rate", 1.0 / unit.picosecond)
        self.addGlobalVariable("thermal_energy", (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature_K * unit.kelvin))
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()
        for name in ["DihedralEnergy","BoostedDihedralEnergy","DihedralBoostPotential","DihedralRefEnergy","DihedralForceScalingFactor",
                     "TotalEnergy","BoostedTotalEnergy","TotalBoostPotential","TotalRefEnergy","TotalForceScalingFactor"]:
            self.addGlobalVariable(name, 0.0)
        self.addGlobalVariable("VminD", float(params.VminD))
        self.addGlobalVariable("VmaxD", float(params.VmaxD))
        self.addGlobalVariable("Dihedralk0", params.k0D)
        self.addGlobalVariable("VminP", float(params.VminP))
        self.addGlobalVariable("VmaxP", float(params.VmaxP))
        self.addGlobalVariable("Totalk0", params.k0P)
        self.addGlobalVariable("refED_factor", params.refED_factor)
        self.addGlobalVariable("refEP_factor", params.refEP_factor)

        self.addComputeGlobal("DihedralEnergy", "energy3")
        self.addComputeGlobal("DihedralRefEnergy", "VminD + (VmaxD - VminD)/max(Dihedralk0, %g)" % _EPS)
        self.beginIfBlock("DihedralRefEnergy > VmaxD + refED_factor*abs(VmaxD)")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.5*Dihedralk0*(DihedralRefEnergy-DihedralEnergy)^2/max(VmaxD-VminD, %g)" % _EPS)
        self.addComputeGlobal("DihedralForceScalingFactor", "1 - Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/max(VmaxD-VminD, %g)" % _EPS)
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedDihedralEnergy", "DihedralEnergy + DihedralBoostPotential")
        self.addComputeGlobal("VminD", "min(DihedralEnergy, VminD)")
        self.addComputeGlobal("VmaxD", "max(DihedralEnergy, VmaxD)")

        self.addComputeGlobal("TotalEnergy", "energy1+energy2+energy3+energy4")
        self.addComputeGlobal("TotalRefEnergy", "VminP + (VmaxP - VminP)/max(Totalk0, %g)" % _EPS)
        self.beginIfBlock("TotalRefEnergy > VmaxP + refEP_factor*abs(VmaxP)")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.endBlock()
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.5*Totalk0*(TotalRefEnergy-TotalEnergy)^2/max(VmaxP-VminP, %g)" % _EPS)
        self.addComputeGlobal("TotalForceScalingFactor", "1 - Totalk0*(TotalRefEnergy-TotalEnergy)/max(VmaxP-VminP, %g)" % _EPS)
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedTotalEnergy", "TotalEnergy + TotalBoostPotential")
        self.addComputeGlobal("VminP", "min(TotalEnergy, VminP)")
        self.addComputeGlobal("VmaxP", "max(TotalEnergy, VmaxP)")

        self.addComputeGlobal("vscale", "exp(-dt*collision_rate)")
        self.addComputeGlobal("fscale", "(1 - vscale)/collision_rate")
        self.addComputeGlobal("noisescale", "sqrt(thermal_energy*(1 - vscale*vscale))")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("v", "v + f0*fscale/m")
        self.addComputePerDof("v", "v + (f1+f2+f4)*TotalForceScalingFactor*fscale/m")
        self.addComputePerDof("v", "v + f3*TotalForceScalingFactor*DihedralForceScalingFactor*fscale/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x - oldx)/dt")

class DualDeepLearningGaMDProduction(CustomIntegrator):
    def __init__(self, dt_ps: float = 0.002, temperature_K: float = 300.0, params: BoostParams | None = None):
        super().__init__(dt_ps * unit.picoseconds)
        if params is None:
            params = BoostParams(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        params.clamp()
        self.addGlobalVariable("collision_rate", 1.0 / unit.picosecond)
        self.addGlobalVariable("thermal_energy", (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature_K * unit.kelvin))
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()
        for name in ["DihedralEnergy","BoostedDihedralEnergy","DihedralBoostPotential","DihedralRefEnergy","DihedralForceScalingFactor",
                     "TotalEnergy","BoostedTotalEnergy","TotalBoostPotential","TotalRefEnergy","TotalForceScalingFactor"]:
            self.addGlobalVariable(name, 0.0)
        self.addGlobalVariable("VminD", float(params.VminD))
        self.addGlobalVariable("VmaxD", float(params.VmaxD))
        self.addGlobalVariable("Dihedralk0", params.k0D)
        self.addGlobalVariable("VminP", float(params.VminP))
        self.addGlobalVariable("VmaxP", float(params.VmaxP))
        self.addGlobalVariable("Totalk0", params.k0P)
        self.addGlobalVariable("refED_factor", params.refED_factor)
        self.addGlobalVariable("refEP_factor", params.refEP_factor)

        self.addComputeGlobal("DihedralEnergy", "energy3")
        self.addComputeGlobal("DihedralRefEnergy", "VminD + (VmaxD - VminD)/max(Dihedralk0, %g)" % _EPS)
        self.beginIfBlock("DihedralRefEnergy > VmaxD + refED_factor*abs(VmaxD)")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.5*Dihedralk0*(DihedralRefEnergy-DihedralEnergy)^2/max(VmaxD-VminD, %g)" % _EPS)
        self.addComputeGlobal("DihedralForceScalingFactor", "1 - Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/max(VmaxD-VminD, %g)" % _EPS)
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedDihedralEnergy", "DihedralEnergy + DihedralBoostPotential")

        self.addComputeGlobal("TotalEnergy", "energy1+energy2+energy3+energy4")
        self.addComputeGlobal("TotalRefEnergy", "VminP + (VmaxP - VminP)/max(Totalk0, %g)" % _EPS)
        self.beginIfBlock("TotalRefEnergy > VmaxP + refEP_factor*abs(VmaxP)")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.endBlock()
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.5*Totalk0*(TotalRefEnergy-TotalEnergy)^2/max(VmaxP-VminP, %g)" % _EPS)
        self.addComputeGlobal("TotalForceScalingFactor", "1 - Totalk0*(TotalRefEnergy-TotalEnergy)/max(VmaxP-VminP, %g)" % _EPS)
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedTotalEnergy", "TotalEnergy + TotalBoostPotential")

        self.addComputeGlobal("vscale", "exp(-dt*collision_rate)")
        self.addComputeGlobal("fscale", "(1 - vscale)/collision_rate")
        self.addComputeGlobal("noisescale", "sqrt(thermal_energy*(1 - vscale*vscale))")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("v", "v + f0*fscale/m")
        self.addComputePerDof("v", "v + (f1+f2+f4)*TotalForceScalingFactor*fscale/m")
        self.addComputePerDof("v", "v + f3*TotalForceScalingFactor*DihedralForceScalingFactor*fscale/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x - oldx)/dt")

def make_conventional(dt_ps: float, temperature_K: float, collision_rate_ps: float = 1.0) -> LangevinMiddleIntegrator:
    return LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        collision_rate_ps / unit.picosecond,
        dt_ps * unit.picoseconds,
    )

def make_dual_equil(dt_ps: float, temperature_K: float, params: BoostParams) -> DualDeepLearningGaMDEquilibration:
    return DualDeepLearningGaMDEquilibration(dt_ps=dt_ps, temperature_K=temperature_K, params=params)

def make_dual_prod(dt_ps: float, temperature_K: float, params: BoostParams) -> DualDeepLearningGaMDProduction:
    return DualDeepLearningGaMDProduction(dt_ps=dt_ps, temperature_K=temperature_K, params=params)
