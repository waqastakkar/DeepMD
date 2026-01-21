"""Build an ACE–ALA–NME alanine dipeptide system in OpenMM."""
from __future__ import annotations

from io import StringIO


def build_alanine_ace_ala_nme_system(solvent: str = "explicit"):
    """Return topology, system, and positions for alanine dipeptide.

    The structure is provided as a minimal ACE–ALA–NME PDB and hydrated with
    hydrogens using the Amber14 force field.
    """
    from openmm import unit
    from openmm.app import (
        ForceField,
        GBn2,
        HBonds,
        Modeller,
        NoCutoff,
        PDBFile,
        PME,
    )

    pdb_text = """
ATOM      1  CH3 ACE A   1      -0.046  -0.001  -0.003  1.00  0.00           C
ATOM      2  C   ACE A   1       1.466   0.000   0.000  1.00  0.00           C
ATOM      3  O   ACE A   1       2.047  -1.135   0.000  1.00  0.00           O
ATOM      4  N   ALA A   2       2.230   1.195   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       3.645   1.416   0.000  1.00  0.00           C
ATOM      6  C   ALA A   2       4.061   0.628   1.246  1.00  0.00           C
ATOM      7  O   ALA A   2       3.337  -0.344   1.534  1.00  0.00           O
ATOM      8  CB  ALA A   2       4.137   2.862   0.000  1.00  0.00           C
ATOM      9  N   NME A   3       5.285   0.984   1.867  1.00  0.00           N
ATOM     10  CH3 NME A   3       5.760   0.203   3.017  1.00  0.00           C
TER
END
"""

    pdb = PDBFile(StringIO(pdb_text))

    solvent_mode = solvent.lower()
    if solvent_mode not in {"explicit", "implicit"}:
        raise ValueError("Solvent mode must be 'explicit' or 'implicit'.")

    def _vector_length(vec):
        if hasattr(vec, "length"):
            return vec.length()
        if hasattr(vec, "norm"):
            return vec.norm()
        try:
            x, y, z = vec
        except TypeError:
            x, y, z = vec.x, vec.y, vec.z
        return (x * x + y * y + z * z) ** 0.5

    if solvent_mode == "explicit":
        forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    else:
        forcefield = ForceField("amber14/protein.ff14SB.xml")

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    if solvent_mode == "explicit":
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=0.8 * unit.nanometer,
        )
        box_vectors = modeller.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise RuntimeError("Explicit solvent requested but no periodic box vectors set.")
        min_box_length = min(_vector_length(vec) for vec in box_vectors)
        max_cutoff = 0.5 * min_box_length - 0.05 * unit.nanometer
        min_cutoff = 0.1 * unit.nanometer
        nonbonded_cutoff = min(1.0 * unit.nanometer, max(min_cutoff, max_cutoff))
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=nonbonded_cutoff,
            constraints=HBonds,
            rigidWater=True,
            ewaldErrorTolerance=1e-4,
        )
    else:
        try:
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=NoCutoff,
                constraints=HBonds,
                implicitSolvent=GBn2,
            )
        except ValueError as exc:
            if "implicitSolvent was specified but never used" in str(exc):
                raise ValueError(
                    "Implicit solvent not supported by selected ForceField XML; "
                    "use explicit or change XML"
                ) from exc
            raise

    positions = modeller.positions
    return modeller.topology, system, positions
