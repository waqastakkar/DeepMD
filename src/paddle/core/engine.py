"""
core/engine.py — OpenMM engine utilities for PADDLE
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

from openmm import Platform, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, Simulation


def is_explicit_simtype(sim_type: str) -> bool:
    return sim_type == "explicit" or sim_type.endswith(".explicit")


@dataclass
class EngineOptions:
    sim_type: str = "RNA.implicit"
    nb_cutoff_angstrom: float = 9.0
    platform_name: str = "CUDA"
    precision: str = "mixed"
    cuda_precision: str = "mixed"
    cuda_device_index: Optional[int] = None
    deterministic_forces: bool = False
    add_barostat: bool = False
    barostat_pressure_atm: float = 1.0
    barostat_interval: int = 25
    ewald_error_tolerance: float = 5e-4
    use_dispersion_correction: bool = True
    rigid_water: bool = True


def get_platform(
    name: str = "CUDA",
    precision: str = "mixed",
    deterministic_forces: bool = False,
    *,
    cuda_precision: Optional[str] = None,
    device_index: Optional[int] = None,
) -> Tuple[Platform, Dict[str, str]]:
    available = {
        Platform.getPlatform(i).getName(): Platform.getPlatform(i)
        for i in range(Platform.getNumPlatforms())
    }
    plat = available.get(name)
    if plat is None:
        if name == "CUDA":
            available_names = ", ".join(sorted(available.keys()))
            raise RuntimeError(
                f"CUDA platform requested but not available. Available platforms: {available_names}"
            )
        plat = available.get("CPU", Platform.getPlatformByName("CPU"))
    props: Dict[str, str] = {}
    plat_name = plat.getName()
    if plat_name in {"CUDA", "OpenCL"}:
        resolved_precision = cuda_precision if plat_name == "CUDA" and cuda_precision else precision
        props["Precision"] = resolved_precision
        props["DeterministicForces"] = "true" if deterministic_forces else "false"
        if device_index is not None:
            props["DeviceIndex"] = str(device_index)
    return plat, props


def build_system(
    parm_file: str,
    sim_type: str,
    nb_cutoff_angstrom: float,
    *,
    ewald_error_tolerance: float,
    use_dispersion_correction: bool,
    rigid_water: bool,
):
    prmtop = AmberPrmtopFile(parm_file)
    if is_explicit_simtype(sim_type):
        from openmm.app import HBonds, PME
        system = prmtop.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=nb_cutoff_angstrom * unit.angstrom,
            constraints=HBonds,
            rigidWater=rigid_water,
            ewaldErrorTolerance=ewald_error_tolerance,
            useDispersionCorrection=use_dispersion_correction,
        )
    else:
        from openmm.app import GBn2
        system = prmtop.createSystem(
            implicitSolvent=GBn2,
            implicitSolventKappa=1.0 / unit.nanometer,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )
    return prmtop, system


def create_simulation(parm_file: str, crd_file: str, integrator, options: EngineOptions):
    prmtop, system = build_system(
        parm_file,
        options.sim_type,
        options.nb_cutoff_angstrom,
        ewald_error_tolerance=options.ewald_error_tolerance,
        use_dispersion_correction=options.use_dispersion_correction,
        rigid_water=options.rigid_water,
    )

    if is_explicit_simtype(options.sim_type) and options.add_barostat:
        from openmm import MonteCarloBarostat
        system.addForce(
            MonteCarloBarostat(
                options.barostat_pressure_atm * unit.atmosphere,
                300.0 * unit.kelvin,
                options.barostat_interval,
            )
        )
        print("Explicit solvent detected → MonteCarloBarostat enabled")

    plat, props = get_platform(
        options.platform_name,
        options.precision,
        options.deterministic_forces,
        cuda_precision=options.cuda_precision,
        device_index=options.cuda_device_index,
    )
    inpcrd = AmberInpcrdFile(crd_file)
    sim = Simulation(prmtop.topology, system, integrator, platform=plat, platformProperties=props)

    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        a, b, c = inpcrd.boxVectors
        sim.context.setPeriodicBoxVectors(a, b, c)

    return sim


def log_simulation_start(
    *,
    stage: str,
    platform_name: str,
    precision: str,
    deterministic_forces: bool,
    dt_ps: float,
    ntcmd: int,
    ntprodpercyc: int,
    explicit: bool,
    rigid_water: bool,
    ewald_error_tolerance: float,
) -> None:
    cmd_ns = (ntcmd * dt_ps) / 1000.0
    prod_ns = (ntprodpercyc * dt_ps) / 1000.0
    print(
        f"[{stage}] platform={platform_name} precision={precision} deterministic={deterministic_forces} "
        f"dt_ps={dt_ps} cmd_ns={cmd_ns:.6g} prod_ns_per_cycle={prod_ns:.6g} "
        f"explicit_pme={explicit} rigidWater={rigid_water} ewaldErrorTolerance={ewald_error_tolerance}"
    )


def minimize_and_initialize(sim: Simulation, temperature_kelvin: float, set_velocities: bool = True) -> None:
    sim.minimizeEnergy()
    if set_velocities:
        sim.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)
