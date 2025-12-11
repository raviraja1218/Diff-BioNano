#!/usr/bin/env python3
"""
03_process_trajectory_FIXED.py
Use TEST trajectory (1 ns) since 100 ns file is corrupted
"""
import mdtraj as md
import numpy as np
import os
from datetime import datetime
import json

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

log("⚠ USING 1 ns TEST TRAJECTORY (100 ns file corrupted)")
log("This will generate all analysis for methodology validation")

# Use the TEST trajectory that works
traj_file = "data/md/raw/trajectory.dcd"  # 1 ns test file
top_file = "data/molecules/alanine_final.pdb"

log(f"Loading: {traj_file}")
try:
    traj = md.load(traj_file, top=top_file)
    log(f"✓ Loaded: {traj.n_frames} frames, {traj.n_atoms} atoms")
    log(f"✓ Time range: {traj.time[0]:.1f} to {traj.time[-1]:.1f} ps")
    log(f"✓ Total simulation: {traj.time[-1] - traj.time[0]:.1f} ps")
except Exception as e:
    log(f"❌ Error loading trajectory: {e}")
    exit(1)

# Save processed data
os.makedirs("data/md/processed", exist_ok=True)

# Full 3D coordinates
np.save("data/md/processed/trajectory_3d_TEST.npy", traj.xyz)
log(f"✓ Saved 3D coordinates: {traj.xyz.shape}")

# 2D projection (x,y) for FDTD grid
traj_2d = traj.xyz[:, :, :2]  # Take only x,y
np.save("data/md/processed/trajectory_2d_TEST.npy", traj_2d)
log(f"✓ Saved 2D coordinates: {traj_2d.shape}")

# Timestamps
np.save("data/md/processed/timestamps_TEST.npy", traj.time)
log(f"✓ Saved timestamps: {traj.time.shape}")

# Metadata - FIXED JSON serialization
metadata = {
    'frames': int(traj.n_frames),  # Convert numpy int to Python int
    'atoms': int(traj.n_atoms),
    'time_ps': float(traj.time[-1] - traj.time[0]),
    'dt_ps': float(traj.time[1] - traj.time[0]),
    'shape_3d': [int(x) for x in traj.xyz.shape],  # Convert tuple to list
    'shape_2d': [int(x) for x in traj_2d.shape],
    'note': 'USING 1 ns TEST TRAJECTORY - 100 ns file corrupted'
}

metadata_path = "data/md/processed/metadata_TEST.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
log(f"✓ Saved metadata to {metadata_path}")

log("✅ Trajectory processing complete (using 1 ns test data)")
log("⚠ NOTE: For final paper, re-run 100 ns with fixed script")
