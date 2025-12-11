#!/usr/bin/env python3
"""
02_production_md.py - 100 ns MD simulation
"""
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import os
from datetime import datetime
import time

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -------------------------------------------------------
# Load minimized structure
# -------------------------------------------------------
log("Loading minimized alanine dipeptide...")
pdb = app.PDBFile("data/molecules/alanine_final.pdb")
log(f"✓ Loaded: {pdb.topology.getNumAtoms()} atoms")
log(f"✓ Residues: {[r.name for r in pdb.topology.residues()]}")

# -------------------------------------------------------
# Create force field and system
# -------------------------------------------------------
log("Creating AMBER99SB-ILDN + GBN2 system...")
forcefield = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    implicitSolvent=app.GBn2
)

# -------------------------------------------------------
# Configure simulation
# -------------------------------------------------------
integrator = mm.LangevinIntegrator(
    300.0 * unit.kelvin,          # Temperature
    1.0 / unit.picosecond,        # Friction coefficient
    0.002 * unit.picoseconds      # 2 fs time step
)

# Use GPU for production
platform = mm.Platform.getPlatformByName("CUDA")
properties = {'Precision': 'mixed'}  # FP32 for speed, sufficient for MD

simulation = app.Simulation(
    pdb.topology, 
    system, 
    integrator, 
    platform,
    properties
)
simulation.context.setPositions(pdb.positions)

# Set initial velocities from Maxwell-Boltzmann distribution
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

# -------------------------------------------------------
# Set up reporters
# -------------------------------------------------------
log("Setting up trajectory and data reporters...")
os.makedirs("data/md/raw", exist_ok=True)

# Save trajectory every 10 ps (5000 steps @ 2 fs)
traj_file = "data/md/raw/trajectory.dcd"
simulation.reporters.append(
    app.DCDReporter(traj_file, 5000)  # Frame every 10 ps
)

# Save simulation state every 1 ps (500 steps)
state_file = "data/md/raw/state.csv"
simulation.reporters.append(
    app.StateDataReporter(
        state_file, 500,  # Report every 1 ps
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=False,  # NVT simulation
        speed=True,    # ns/day
        separator=','
    )
)

# Also print progress to console every 10 ps
simulation.reporters.append(
    app.StateDataReporter(
        None, 5000,  # Console every 10 ps
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True
    )
)

# -------------------------------------------------------
# Run production MD
# -------------------------------------------------------
# 100 ns = 100,000 ps = 50,000,000 steps @ 2 fs
total_steps = 50000000
log(f"Starting 100 ns production MD ({total_steps:,} steps)...")
log(f"Estimated time on RTX 4050: ~4-6 hours")
log(f"Trajectory will be saved to: {traj_file}")
log(f"State data: {state_file}")

start_time = time.time()

# Run in chunks for better progress tracking
chunk_size = 1000000  # 2 ns chunks
num_chunks = total_steps // chunk_size

for chunk in range(num_chunks):
    chunk_start = time.time()
    
    simulation.step(chunk_size)
    
    chunk_time = time.time() - chunk_start
    completed_steps = (chunk + 1) * chunk_size
    completed_time_ps = completed_steps * 0.002  # 2 fs per step
    
    log(f"Chunk {chunk+1}/{num_chunks} complete: {completed_time_ps:.1f} ps "
        f"({(chunk+1)*2} ns) in {chunk_time:.1f}s")
    
    # Estimate remaining time
    if chunk > 0:
        avg_time_per_chunk = (time.time() - start_time) / (chunk + 1)
        remaining_chunks = num_chunks - (chunk + 1)
        remaining_time = avg_time_per_chunk * remaining_chunks
        log(f"Estimated remaining: {remaining_time/60:.1f} minutes")

# -------------------------------------------------------
# Completion
# -------------------------------------------------------
total_time = time.time() - start_time
log(f"\n✅ 100 ns MD SIMULATION COMPLETE!")
log(f"Total simulation time: {total_time/60:.1f} minutes")
log(f"Performance: {50/total_time*3600:.1f} ns/day")  # 50M steps = 100 ns

# Save final structure
final_positions = simulation.context.getState(getPositions=True).getPositions()
final_pdb = "data/md/raw/final_structure.pdb"
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(pdb.topology, final_positions, f)
log(f"✓ Final structure saved: {final_pdb}")

# -------------------------------------------------------
# Quick validation
# -------------------------------------------------------
log("\n=== VALIDATION ===")

# Count frames in trajectory
import mdtraj as md
try:
    traj = md.load(traj_file, top=pdb.topology)
    log(f"Trajectory frames: {traj.n_frames}")
    log(f"Time span: {traj.time[0]:.1f} to {traj.time[-1]:.1f} ps")
except:
    log("⚠ Could not load trajectory with mdtraj (install later)")

# Check state file
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        lines = f.readlines()
        log(f"State data points: {len(lines)-1}")  # minus header
else:
    log("⚠ State file not found")

log("\n✅ PHASE 2 MD SIMULATION COMPLETE")
log("✅ Proceed to trajectory analysis")
