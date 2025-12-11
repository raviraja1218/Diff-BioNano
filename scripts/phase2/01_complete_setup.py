#!/usr/bin/env python3
"""
01_setup_simulation_COMPLETE.py
Creates complete alanine dipeptide with ALL hydrogens
"""
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app import PDBFile
import os
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -------------------------------------------------------
# METHOD 1: Use OpenMM's built-in test system creator
# -------------------------------------------------------
log("Creating alanine dipeptide with ALL atoms...")

try:
    # This is the MAGIC function - creates complete ACE-ALA-NME
    from openmm.app.internal import createAlanineDipeptide
    
    # Create topology and positions
    topology, positions = createAlanineDipeptide()
    
    log(f"✓ Created via OpenMM internal: {topology.getNumAtoms()} atoms")
    
except ImportError:
    log("OpenMM internal method not available, using alternative...")
    
    # Alternative: Create using Modeller
    from openmm.app import Modeller, Topology
    from openmm.app.element import Element
    
    # Create topology
    topology = Topology()
    chain = topology.addChain()
    
    # Add ACE residue with ALL atoms
    ace = topology.addResidue("ACE", chain)
    topology.addAtom("CH3", Element.getBySymbol("C"), ace)  # Methyl carbon
    topology.addAtom("HH31", Element.getBySymbol("H"), ace)  # H
    topology.addAtom("HH32", Element.getBySymbol("H"), ace)  # H  
    topology.addAtom("HH33", Element.getBySymbol("H"), ace)  # H
    topology.addAtom("C", Element.getBySymbol("C"), ace)     # Carbonyl C
    topology.addAtom("O", Element.getBySymbol("O"), ace)     # Carbonyl O
    
    # Add ALA residue
    ala = topology.addResidue("ALA", chain)
    topology.addAtom("N", Element.getBySymbol("N"), ala)
    topology.addAtom("H", Element.getBySymbol("H"), ala)     # Amide H
    topology.addAtom("CA", Element.getBySymbol("C"), ala)
    topology.addAtom("HA", Element.getBySymbol("H"), ala)    # Alpha H
    topology.addAtom("CB", Element.getBySymbol("C"), ala)
    topology.addAtom("HB1", Element.getBySymbol("H"), ala)   # Beta H
    topology.addAtom("HB2", Element.getBySymbol("H"), ala)
    topology.addAtom("HB3", Element.getBySymbol("H"), ala)
    topology.addAtom("C", Element.getBySymbol("C"), ala)
    topology.addAtom("O", Element.getBySymbol("O"), ala)
    
    # Add NME residue
    nme = topology.addResidue("NME", chain)
    topology.addAtom("N", Element.getBySymbol("N"), nme)
    topology.addAtom("H", Element.getBySymbol("H"), nme)     # Amide H
    topology.addAtom("CH3", Element.getBySymbol("C"), nme)   # Methyl C
    topology.addAtom("HH31", Element.getBySymbol("H"), nme)  # H
    topology.addAtom("HH32", Element.getBySymbol("H"), nme)  # H
    topology.addAtom("HH33", Element.getBySymbol("H"), nme)  # H
    
    # Create approximate positions
    import numpy as np
    positions = []
    # ACE coordinates
    ace_coords = [
        [-1.141, 0.616, 0.000],    # CH3
        [-1.641, 0.996, 0.890],    # HH31
        [-1.641, 1.136, -0.890],   # HH32  
        [-1.641, -0.464, 0.000],   # HH33
        [-0.099, -0.374, 0.000],   # C
        [-0.169, -1.593, 0.000],   # O
    ]
    positions.extend([pos * unit.angstroms for pos in ace_coords])
    
    # ALA coordinates
    ala_coords = [
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
    ]
    positions.extend([pos * unit.angstroms for pos in ala_coords])
    
    # NME coordinates
    nme_coords = [
        [3.239, 1.492, 0.000],     # N
        [2.789, 2.272, 0.000],     # H
        [4.388, 2.292, 0.000],     # CH3
        [4.888, 2.692, 0.890],     # HH31
        [4.888, 2.892, -0.890],    # HH32
        [4.888, 1.492, 0.000],     # HH33
    ]
    positions.extend([pos * unit.angstroms for pos in nme_coords])

# Save initial structure
os.makedirs("data/molecules", exist_ok=True)
initial_pdb = "data/molecules/alanine_initial.pdb"
with open(initial_pdb, "w") as f:
    PDBFile.writeFile(topology, positions, f)

log(f"✓ Initial structure saved: {initial_pdb}")
log(f"✓ Total atoms: {len(positions)}")

# -------------------------------------------------------
# Build force field with MODELLER (adds missing hydrogens)
# -------------------------------------------------------
log("Using Modeller to add missing hydrogens...")

# Load from PDB to ensure clean topology
pdb = PDBFile(initial_pdb)

# Create force field
forcefield = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")

# Use Modeller to add hydrogens properly
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield)

log(f"✓ After adding hydrogens: {modeller.topology.getNumAtoms()} atoms")

# Create system from modeller topology
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    implicitSolvent=app.GBn2
)

# -------------------------------------------------------
# Create simulation
# -------------------------------------------------------
integrator = mm.LangevinIntegrator(
    300.0 * unit.kelvin,
    1.0 / unit.picosecond,
    0.002 * unit.picoseconds
)

# Use CPU for stability (then switch to GPU for production)
platform = mm.Platform.getPlatformByName("CPU")

simulation = app.Simulation(
    modeller.topology, 
    system, 
    integrator, 
    platform
)
simulation.context.setPositions(modeller.positions)

# -------------------------------------------------------
# Minimize energy
# -------------------------------------------------------
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
# Verify structure
# -------------------------------------------------------
from openmm.app import PDBFile
final_pdb = PDBFile(output_path)
log(f"\n=== FINAL STRUCTURE ===")
log(f"Atoms: {final_pdb.topology.getNumAtoms()}")
log(f"Residues: {[r.name for r in final_pdb.topology.residues()]}")

# Count atoms per residue
for residue in final_pdb.topology.residues():
    atoms = list(residue.atoms())
    log(f"  {residue.name}: {len(atoms)} atoms")

log("✓ Setup complete. Ready for production MD.")
