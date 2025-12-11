#!/usr/bin/env python3
"""
01_final_setup.py - FIXED unit handling
"""
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app import PDBFile
import os
import numpy as np
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -------------------------------------------------------
# Create topology WITHOUT units in initial positions
# -------------------------------------------------------
log("Creating alanine dipeptide topology...")

# Create a simple topology with correct residues
from openmm.app import Topology
from openmm.app.element import Element

topology = Topology()
chain = topology.addChain()

# Add ACE residue
ace = topology.addResidue("ACE", chain)
atoms_ace = [
    topology.addAtom("CH3", Element.getBySymbol("C"), ace),
    topology.addAtom("HH31", Element.getBySymbol("H"), ace),
    topology.addAtom("HH32", Element.getBySymbol("H"), ace),
    topology.addAtom("HH33", Element.getBySymbol("H"), ace),
    topology.addAtom("C", Element.getBySymbol("C"), ace),
    topology.addAtom("O", Element.getBySymbol("O"), ace),
]

# Add ALA residue  
ala = topology.addResidue("ALA", chain)
atoms_ala = [
    topology.addAtom("N", Element.getBySymbol("N"), ala),
    topology.addAtom("H", Element.getBySymbol("H"), ala),
    topology.addAtom("CA", Element.getBySymbol("C"), ala),
    topology.addAtom("HA", Element.getBySymbol("H"), ala),
    topology.addAtom("CB", Element.getBySymbol("C"), ala),
    topology.addAtom("HB1", Element.getBySymbol("H"), ala),
    topology.addAtom("HB2", Element.getBySymbol("H"), ala),
    topology.addAtom("HB3", Element.getBySymbol("H"), ala),
    topology.addAtom("C", Element.getBySymbol("C"), ala),
    topology.addAtom("O", Element.getBySymbol("O"), ala),
]

# Add NME residue
nme = topology.addResidue("NME", chain)
atoms_nme = [
    topology.addAtom("N", Element.getBySymbol("N"), nme),
    topology.addAtom("H", Element.getBySymbol("H"), nme),
    topology.addAtom("CH3", Element.getBySymbol("C"), nme),
    topology.addAtom("HH31", Element.getBySymbol("H"), nme),
    topology.addAtom("HH32", Element.getBySymbol("H"), nme),
    topology.addAtom("HH33", Element.getBySymbol("H"), nme),
]

# Add bonds
for i in range(3):  # CH3 hydrogens
    topology.addBond(atoms_ace[0], atoms_ace[i+1])
topology.addBond(atoms_ace[0], atoms_ace[4])  # CH3-C
topology.addBond(atoms_ace[4], atoms_ace[5])  # C=O
topology.addBond(atoms_ace[4], atoms_ala[0])  # ACE C - ALA N (peptide)

# ALA bonds
topology.addBond(atoms_ala[0], atoms_ala[1])  # N-H
topology.addBond(atoms_ala[0], atoms_ala[2])  # N-CA
topology.addBond(atoms_ala[2], atoms_ala[3])  # CA-HA
topology.addBond(atoms_ala[2], atoms_ala[4])  # CA-CB
for i in range(3):  # CB hydrogens
    topology.addBond(atoms_ala[4], atoms_ala[5+i])
topology.addBond(atoms_ala[2], atoms_ala[8])  # CA-C
topology.addBond(atoms_ala[8], atoms_ala[9])  # C=O
topology.addBond(atoms_ala[8], atoms_nme[0])  # ALA C - NME N

# NME bonds
topology.addBond(atoms_nme[0], atoms_nme[1])  # N-H
topology.addBond(atoms_nme[0], atoms_nme[2])  # N-CH3
for i in range(3):  # CH3 hydrogens
    topology.addBond(atoms_nme[2], atoms_nme[3+i])

# -------------------------------------------------------
# Create initial positions WITHOUT units
# -------------------------------------------------------
log("Creating initial coordinates...")

# Create positions in angstroms (no units yet)
positions_angstrom = np.array([
    # ACE
    [-1.141, 0.616, 0.000],    # CH3
    [-1.641, 0.996, 0.890],    # HH31
    [-1.641, 1.136, -0.890],   # HH32  
    [-1.641, -0.464, 0.000],   # HH33
    [-0.099, -0.374, 0.000],   # C
    [-0.169, -1.593, 0.000],   # O
    # ALA
    [0.995, 0.197, 0.000],     # N
    [1.445, 0.677, 0.000],     # H
    [2.102, -0.617, 0.000],    # CA
    [2.602, -0.717, 0.890],    # HA
    [2.381, -1.206, 1.399],    # CB
    [1.881, -1.706, 1.889],    # HB1
    [3.381, -1.706, 1.889],    # HB2
    [2.381, -0.706, 2.399],    # HB3
    [3.303, 0.173, 0.000],     # C
    [4.257, -0.587, 0.000],    # O
    # NME
    [3.239, 1.492, 0.000],     # N
    [2.789, 2.272, 0.000],     # H
    [4.388, 2.292, 0.000],     # CH3
    [4.888, 2.692, 0.890],     # HH31
    [4.888, 2.892, -0.890],    # HH32
    [4.888, 1.492, 0.000],     # HH33
], dtype=np.float32)

# Save initial PDB WITHOUT units
os.makedirs("data/molecules", exist_ok=True)
initial_pdb = "data/molecules/alanine_initial.pdb"

# Write PDB manually to avoid unit issues
with open(initial_pdb, "w") as f:
    f.write("REMARK   Alanine Dipeptide (ACE-ALA-NME) - Initial\n")
    f.write("REMARK   Created for Diff-BioNano project\n")
    
    atom_count = 1
    atom_names = ["CH3", "HH31", "HH32", "HH33", "C", "O",
                  "N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O",
                  "N", "H", "CH3", "HH31", "HH32", "HH33"]
    res_names = ["ACE"]*6 + ["ALA"]*10 + ["NME"]*6
    
    for i in range(22):
        x, y, z = positions_angstrom[i]
        f.write(f"ATOM  {atom_count:6d} {atom_names[i]:4s} {res_names[i]:3s} A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_names[i][0]}\n")
        atom_count += 1
    
    f.write("TER\n")
    f.write("END\n")

log(f"✓ Initial structure saved: {initial_pdb}")

# -------------------------------------------------------
# Load with OpenMM and add hydrogens properly
# -------------------------------------------------------
log("Loading with OpenMM and adding hydrogens...")

pdb = PDBFile(initial_pdb)
log(f"✓ Loaded: {pdb.topology.getNumAtoms()} atoms")

# Now add units to positions for OpenMM
positions = positions_angstrom * unit.angstroms

# Create force field
forcefield = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")

# Use Modeller to ensure complete structure
modeller = app.Modeller(pdb.topology, positions)
modeller.addHydrogens(forcefield)

log(f"✓ After adding hydrogens: {modeller.topology.getNumAtoms()} atoms")

# Create system
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    implicitSolvent=app.GBn2
)

# -------------------------------------------------------
# Minimize energy
# -------------------------------------------------------
integrator = mm.LangevinIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds
)

# Use CPU for stability
platform = mm.Platform.getPlatformByName("CPU")

simulation = app.Simulation(
    modeller.topology, 
    system, 
    integrator, 
    platform
)
simulation.context.setPositions(modeller.positions)

log("Energy minimizing...")
simulation.minimizeEnergy(maxIterations=1000)

state = simulation.context.getState(getPositions=True, getEnergy=True)
min_pos = state.getPositions()
energy = state.getPotentialEnergy()

log(f"✓ Minimization complete. Energy: {energy}")

# -------------------------------------------------------
# Save minimized structure
# -------------------------------------------------------
output_path = "data/molecules/alanine_dipeptide_minimized.pdb"
with open(output_path, "w") as f:
    PDBFile.writeFile(modeller.topology, min_pos, f)

log(f"✓ Final structure saved: {output_path}")

# -------------------------------------------------------
# Verification
# -------------------------------------------------------
final_pdb = PDBFile(output_path)
log(f"\n=== FINAL VERIFICATION ===")
log(f"Total atoms: {final_pdb.topology.getNumAtoms()}")
log(f"Residues: {[r.name for r in final_pdb.topology.residues()]}")

# Quick visualization of first few atoms
log("\nFirst 5 atoms:")
for i, atom in enumerate(list(final_pdb.topology.atoms())[:5]):
    res = list(atom.residue())[0] if atom.residue() else None
    log(f"  Atom {i+1}: {atom.name} in {res.name if res else 'N/A'}")

log("\n✅ PHASE 2 SETUP COMPLETE")
log("✅ Ready for 100 ns MD simulation")
