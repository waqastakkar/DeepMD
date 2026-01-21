"""Build an ACE–ALA–NME alanine dipeptide system in OpenMM."""
from __future__ import annotations

from io import StringIO


def build_alanine_ace_ala_nme_system():
    """Return topology, system, and positions for alanine dipeptide.

    The structure is provided as a minimal ACE–ALA–NME PDB and hydrated with
    hydrogens using the Amber14 force field. The system is configured for
    implicit solvent with GBn2 and no cutoff for long-range interactions.
    """
    from openmm.app import ForceField, GBn2, HBonds, Modeller, NoCutoff, PDBFile

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
    forcefield = ForceField("amber14/protein.ff14SB.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        implicitSolvent=GBn2,
    )

    positions = modeller.positions
    return modeller.topology, system, positions
