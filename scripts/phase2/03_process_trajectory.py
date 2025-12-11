#!/usr/bin/env python3
"""
03_process_trajectory.py
Convert DCD to numpy arrays for analysis.
"""
import mdtraj as md
import numpy as np
import os
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

log("Processing 100 ns trajectory...")

# Load the production trajectory
traj = md.load("data/md/raw/trajectory_100ns.dcd",
               top="data/molecules/alanine_final.pdb")

log(f"✓ Loaded: {traj.n_frames} frames, {traj.n_atoms} atoms")
log(f"✓ Time range: {traj.time[0]:.1f} to {traj.time[-1]:.1f} ps")

# Save processed data
os.makedirs("data/md/processed", exist_ok=True)

# Full 3D coordinates (frames, atoms, xyz)
np.save("data/md/processed/trajectory_3d.npy", traj.xyz)
log(f"✓ Saved 3D coordinates: {traj.xyz.shape}")

# 2D projection (x,y) for FDTD grid
traj_2d = traj.xyz[:, :, :2]  # Take only x,y
np.save("data/md/processed/trajectory_2d.npy", traj_2d)
log(f"✓ Saved 2D coordinates: {traj_2d.shape}")

# Timestamps
np.save("data/md/processed/timestamps.npy", traj.time)
log(f"✓ Saved timestamps: {traj.time.shape}")

# Metadata
import json
metadata = {
    'frames': traj.n_frames,
    'atoms': traj.n_atoms,
    'time_ps': traj.time[-1] - traj.time[0],
    'dt_ps': traj.time[1] - traj.time[0],
    'shape_3d': traj.xyz.shape,
    'shape_2d': traj_2d.shape
}
with open("data/md/processed/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
log("✓ Saved metadata")

log("✅ Trajectory processing complete!")
