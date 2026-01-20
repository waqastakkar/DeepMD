"""
core/engine.py — OpenMM engine utilities for PADDLE
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import os

from openmm import Platform, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, Simulation


@dataclass
class EngineOptions:
    sim_type: str = "RNA.implicit"
    nb_cutoff_angstrom: float = 9.0
    platform_name: str = "CUDA"
    precision: str = "mixed"
    deterministic_forces: bool = False
    add_barostat: bool = False
    barostat_pressure_atm: float = 1.0
    barostat_interval: int = 25


def get_platform(name: str = "CUDA", precision: str = "mixed", deterministic_forces: bool = False) -> Tuple[Platform, Dict[str, str]]:
    plat = Platform.getPlatformByName(name)
    props: Dict[str, str] = {}
    if name in {"CUDA", "OpenCL"}:
        props["Precision"] = precision
        if deterministic_forces:
            props["DeterministicForces"] = "true"
    return plat, props


def build_system(parm_file: str, sim_type: str, nb_cutoff_angstrom: float):
    prmtop = AmberPrmtopFile(parm_file)
    if sim_type == "explicit":
        from openmm.app import HBonds, PME
        system = prmtop.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=nb_cutoff_angstrom * unit.angstrom,
            constraints=HBonds,
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
    prmtop, system = build_system(parm_file, options.sim_type, options.nb_cutoff_angstrom)

    if options.sim_type == "explicit" and options.add_barostat:
        from openmm import MonteCarloBarostat
        system.addForce(
            MonteCarloBarostat(
                options.barostat_pressure_atm * unit.atmosphere,
                300.0 * unit.kelvin,
                options.barostat_interval,
            )
        )

    plat, props = get_platform(options.platform_name, options.precision, options.deterministic_forces)
    inpcrd = AmberInpcrdFile(crd_file)
    sim = Simulation(prmtop.topology, system, integrator, platform=plat, platformProperties=props)

    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        a, b, c = inpcrd.boxVectors
        sim.context.setPeriodicBoxVectors(a, b, c)

    return sim


def minimize_and_initialize(sim: Simulation, temperature_kelvin: float, set_velocities: bool = True) -> None:
    sim.minimizeEnergy()
    if set_velocities:
        sim.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)
