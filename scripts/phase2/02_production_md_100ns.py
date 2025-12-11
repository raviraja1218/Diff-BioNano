#!/usr/bin/env python3
"""
02_production_md_100ns.py - 100 ns PRODUCTION MD simulation
"""
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import os
import sys
from datetime import datetime
import time

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -------------------------------------------------------
# Load minimized structure
# -------------------------------------------------------
log("Loading minimized alanine dipeptide...")
pdb_path = "data/molecules/alanine_final.pdb"
if not os.path.exists(pdb_path):
    log(f"❌ ERROR: {pdb_path} not found!")
    sys.exit(1)

pdb = app.PDBFile(pdb_path)
log(f"✓ Loaded: {pdb.topology.getNumAtoms()} atoms")
log(f"✓ Residues: {[r.name for r in pdb.topology.residues()]}")

# -------------------------------------------------------
# Create force field and system
# -------------------------------------------------------
log("Creating AMBER99SB-ILDN + GBN2 system...")
forcefield = app.ForceField("amber99sbildn.xml", "implicit/gbn2.xml")

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,  # Implicit solvent
    constraints=app.HBonds
)

log(f"✓ System created with {system.getNumParticles()} particles")

# -------------------------------------------------------
# Configure simulation
# -------------------------------------------------------
integrator = mm.LangevinIntegrator(
    300.0 * unit.kelvin,          # Temperature
    1.0 / unit.picosecond,        # Friction coefficient
    0.002 * unit.picoseconds      # 2 fs time step
)

# Use GPU for production
try:
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {'Precision': 'mixed', 'DeviceIndex': '0'}  # Use first GPU
    log("✓ Using CUDA GPU platform (RTX 4050)")
except Exception as e:
    log(f"⚠ GPU not available: {e}")
    platform = mm.Platform.getPlatformByName("CPU")
    properties = {}
    log("⚠ Using CPU (slower)")

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
traj_file = "data/md/raw/trajectory_100ns.dcd"
simulation.reporters.append(
    app.DCDReporter(traj_file, 5000)  # Frame every 10 ps
)

# Save simulation state every 10 ps (5000 steps) - less frequent to reduce file size
state_file = "data/md/raw/state_100ns.csv"
simulation.reporters.append(
    app.StateDataReporter(
        state_file, 5000,  # Report every 10 ps
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=False,
        speed=True,    # ns/day
        separator=',',
        totalSteps=50000000
    )
)

# Also print progress to console every 100 ps (50,000 steps)
simulation.reporters.append(
    app.StateDataReporter(
        None, 50000,  # Console every 100 ps (0.1 ns)
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=50000000
    )
)

# -------------------------------------------------------
# Run 100 ns PRODUCTION MD
# -------------------------------------------------------
# 100 ns = 100,000 ps = 50,000,000 steps @ 2 fs
total_steps = 50000000
log(f"🚀 STARTING 100 ns PRODUCTION MD")
log(f"Total steps: {total_steps:,}")
log(f"Trajectory frames: {total_steps//5000:,} (every 10 ps)")
log(f"Trajectory file: {traj_file}")
log(f"State data: {state_file}")

# Performance estimate based on 1 ns test
estimated_hours = 1.5  # Based on 1 ns = 55 seconds
log(f"Estimated time: ~{estimated_hours:.1f} hours (based on 1 ns test)")

start_time = time.time()
log(f"Simulation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run in chunks for progress tracking and safety
chunk_size = 1000000  # 2 ns chunks
num_chunks = total_steps // chunk_size
log(f"Running in {num_chunks} chunks of {chunk_size:,} steps (2 ns each)")

for chunk in range(num_chunks):
    chunk_start = time.time()
    
    # Run chunk
    simulation.step(chunk_size)
    
    # Calculate statistics
    chunk_time = time.time() - chunk_start
    completed_steps = (chunk + 1) * chunk_size
    completed_ns = completed_steps * 0.002 / 1000  # Convert to ns
    
    # Progress percentage
    progress_pct = (completed_steps / total_steps) * 100
    
    log(f"✅ Chunk {chunk+1}/{num_chunks} complete")
    log(f"   Progress: {progress_pct:.1f}% ({completed_ns:.1f} / 100.0 ns)")
    log(f"   Chunk time: {chunk_time:.1f}s")
    
    # Running average and estimate
    if chunk > 0:
        avg_time_per_chunk = (time.time() - start_time) / (chunk + 1)
        remaining_chunks = num_chunks - (chunk + 1)
        remaining_seconds = avg_time_per_chunk * remaining_chunks
        
        # Format remaining time
        if remaining_seconds < 60:
            remaining_str = f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            remaining_str = f"{remaining_seconds/60:.1f}m"
        else:
            remaining_str = f"{remaining_seconds/3600:.1f}h"
        
        # Current speed
        current_speed = (chunk_size * 0.002) / (chunk_time / 1e6)  # ns/day
        log(f"   Current speed: {current_speed:,.0f} ns/day")
        log(f"   Estimated remaining: {remaining_str}")
    
    # Save checkpoint every 5 chunks (10 ns)
    if (chunk + 1) % 5 == 0:
        checkpoint_time = time.time() - start_time
        log(f"   🎯 Checkpoint: {completed_ns:.1f} ns in {checkpoint_time/60:.1f} minutes")
        
        # Quick file size check
        if os.path.exists(traj_file):
            size_mb = os.path.getsize(traj_file) / (1024*1024)
            log(f"   Trajectory size: {size_mb:.1f} MB")

# -------------------------------------------------------
# COMPLETION
# -------------------------------------------------------
total_time = time.time() - start_time
total_minutes = total_time / 60
total_hours = total_minutes / 60

log(f"\n{'='*60}")
log(f"🎉 100 ns MD SIMULATION COMPLETE!")
log(f"{'='*60}")
log(f"Total simulation time: {total_time:.1f}s ({total_minutes:.1f} min, {total_hours:.2f} hr)")
log(f"Final performance: {total_steps*0.002/total_time*1e6:,.0f} ns/day")
log(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Save final structure
final_positions = simulation.context.getState(getPositions=True).getPositions()
final_pdb = "data/md/raw/final_structure_100ns.pdb"
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(pdb.topology, final_positions, f)
log(f"✓ Final structure saved: {final_pdb}")

# -------------------------------------------------------
# VALIDATION
# -------------------------------------------------------
log(f"\n{'='*60}")
log("VALIDATION CHECK")
log(f"{'='*60}")

# Check all output files
files_to_check = [
    (traj_file, "Trajectory (DCD)"),
    (state_file, "State data (CSV)"),
    (final_pdb, "Final structure (PDB)")
]

all_good = True
for filepath, description in files_to_check:
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        log(f"✓ {description}: {size_mb:.1f} MB")
        
        # Quick sanity check for trajectory
        if filepath.endswith('.dcd') and size_mb < 1:
            log(f"⚠ WARNING: Trajectory file seems too small ({size_mb:.1f} MB)")
            all_good = False
    else:
        log(f"❌ {description}: FILE NOT FOUND!")
        all_good = False

# Check state file has expected lines (should have ~10000 lines for 100 ns @ 10 ps)
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        lines = f.readlines()
        data_points = len(lines) - 1  # Minus header
        expected_points = 10000  # 100 ns / 10 ps
        log(f"✓ State data points: {data_points} (expected: ~{expected_points})")
        
        if data_points < expected_points * 0.9:  # Allow 10% tolerance
            log(f"⚠ WARNING: Fewer data points than expected")
            all_good = False

# Final status
if all_good:
    log(f"\n✅ ALL CHECKS PASSED!")
    log("✅ 100 ns MD simulation SUCCESSFUL!")
    log("✅ Ready for trajectory analysis")
else:
    log(f"\n⚠ WARNING: Some checks failed")
    log("⚠ Review output files manually")

log(f"\n{'='*60}")
log("NEXT STEPS:")
log("1. Run trajectory processing: python scripts/phase2/03_process_trajectory.py")
log("2. Generate Ramachandran plot: python scripts/phase2/04_ramachandran_plot.py")
log("3. Generate RMSD plot: python scripts/phase2/05_rmsd_convergence.py")
log("4. Generate density plot: python scripts/phase2/06_probability_density.py")
log("5. Create statistics table: python scripts/phase2/07_md_statistics_table.py")
log(f"{'='*60}")

# Save completion marker
with open("data/md/raw/SIMULATION_COMPLETE.txt", "w") as f:
    f.write(f"100 ns MD simulation completed successfully\n")
    f.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total time: {total_time:.1f} seconds\n")
    f.write(f"Performance: {total_steps*0.002/total_time*1e6:.0f} ns/day\n")

log("✓ Completion marker saved")
