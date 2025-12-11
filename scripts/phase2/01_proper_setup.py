#!/usr/bin/env python3
"""
01_proper_setup.py - Correct PDB format
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
# Create PROPER PDB format (columns matter!)
# -------------------------------------------------------
log("Creating alanine dipeptide with correct PDB format...")

# PROPER PDB format with exact columns
# Columns: 1-6: "ATOM  ", 7-11: atom number, 13-16: atom name, 18-20: residue name
# 22: chain, 23-26: residue number, 31-38: x, 39-46: y, 47-54: z
pdb_content = """\
ATOM      1  CH3 ACE A   1      -1.141   0.616   0.000  1.00  0.00           C
ATOM      2 HH31 ACE A   1      -1.641   0.996   0.890  1.00  0.00           H
ATOM      3 HH32 ACE A   1      -1.641   1.136  -0.890  1.00  0.00           H
ATOM      4 HH33 ACE A   1      -1.641  -0.464   0.000  1.00  0.00           H
ATOM      5    C ACE A   1      -0.099  -0.374   0.000  1.00  0.00           C
ATOM      6    O ACE A   1      -0.169  -1.593   0.000  1.00  0.00           O
ATOM      7    N ALA A   2       0.995   0.197   0.000  1.00  0.00           N
ATOM      8    H ALA A   2       1.445   0.677   0.000  1.00  0.00           H
ATOM      9   CA ALA A   2       2.102  -0.617   0.000  1.00  0.00           C
ATOM     10   HA ALA A   2       2.602  -0.717   0.890  1.00  0.00           H
ATOM     11   CB ALA A   2       2.381  -1.206   1.399  1.00  0.00           C
ATOM     12  HB1 ALA A   2       1.881  -1.706   1.889  1.00  0.00           H
ATOM     13  HB2 ALA A   2       3.381  -1.706   1.889  1.00  0.00           H
ATOM     14  HB3 ALA A   2       2.381  -0.706   2.399  1.00  0.00           H
ATOM     15    C ALA A   2       3.303   0.173   0.000  1.00  0.00           C
ATOM     16    O ALA A   2       4.257  -0.587   0.000  1.00  0.00           O
ATOM     17    N NME A   3       3.239   1.492   0.000  1.00  0.00           N
ATOM     18    H NME A   3       2.789   2.272   0.000  1.00  0.00           H
ATOM     19  CH3 NME A   3       4.388   2.292   0.000  1.00  0.00           C
ATOM     20 HH31 NME A   3       4.888   2.692   0.890  1.00  0.00           H
ATOM     21 HH32 NME A   3       4.888   2.892  -0.890  1.00  0.00           H
ATOM     22 HH33 NME A   3       4.888   1.492   0.000  1.00  0.00           H
TER
"""

# Save to file
os.makedirs("data/molecules", exist_ok=True)
initial_pdb = "data/molecules/alanine_proper.pdb"
with open(initial_pdb, "w") as f:
    f.write(pdb_content)

log(f"✓ Saved properly formatted PDB: {initial_pdb}")

# -------------------------------------------------------
# Load and process
# -------------------------------------------------------
log("Loading PDB...")
pdb = PDBFile(initial_pdb)
log(f"✓ Loaded: {pdb.topology.getNumAtoms()} atoms")
log(f"✓ Residues: {[r.name for r in pdb.topology.residues()]}")

# Check atom names
log("\nAtom names in first residue:")
first_res = list(pdb.topology.residues())[0]
for atom in first_res.atoms():
    log(f"  {atom.name}")

# -------------------------------------------------------
# Add missing hydrogens using Modeller
# -------------------------------------------------------
log("\nAdding missing hydrogens with Modeller...")

# Try with different force fields
try:
    forcefield = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")
    log("✓ Using AMBER99SB-ILDN + GBN2")
except:
    forcefield = app.ForceField("amber14-all.xml")
    log("✓ Using AMBER14-ALL")

# Use Modeller
modeller = app.Modeller(pdb.topology, pdb.positions)
log(f"Before adding hydrogens: {modeller.topology.getNumAtoms()} atoms")

try:
    modeller.addHydrogens(forcefield)
    log(f"After adding hydrogens: {modeller.topology.getNumAtoms()} atoms")
except Exception as e:
    log(f"⚠ Could not add hydrogens: {e}")
    log("Continuing with existing structure...")

# Create system
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds
)

# -------------------------------------------------------
# Minimize
# -------------------------------------------------------
integrator = mm.LangevinIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds
)

# Use CPU for reliability
platform = mm.Platform.getPlatformByName("CPU")

simulation = app.Simulation(
    modeller.topology, 
    system, 
    integrator, 
    platform
)
simulation.context.setPositions(modeller.positions)

log("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=1000)

state = simulation.context.getState(getPositions=True, getEnergy=True)
min_pos = state.getPositions()
energy = state.getPotentialEnergy()

log(f"✓ Minimization complete. Energy: {energy}")

# -------------------------------------------------------
# Save final structure
# -------------------------------------------------------
output_pdb = "data/molecules/alanine_final.pdb"
with open(output_pdb, "w") as f:
    PDBFile.writeFile(modeller.topology, min_pos, f)

log(f"✓ Final structure saved: {output_pdb}")

# -------------------------------------------------------
# Verify
# -------------------------------------------------------
final = PDBFile(output_pdb)
log(f"\n=== FINAL STRUCTURE ===")
log(f"Atoms: {final.topology.getNumAtoms()}")
log(f"Residues: {[r.name for r in final.topology.residues()]}")

# Show first 3 atoms
log("\nFirst 3 atoms:")
for i, atom in enumerate(list(final.topology.atoms())[:3]):
    res = list(atom.residue())[0]
    log(f"  {atom.name} in {res.name}")

log("\n✅ SETUP COMPLETE")
log("✅ Ready for MD simulation")
