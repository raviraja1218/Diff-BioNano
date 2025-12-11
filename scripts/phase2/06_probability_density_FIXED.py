#!/usr/bin/env python3
"""
06_probability_density_FIXED.py
Use TEST trajectory data
"""
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating probability density map from 1 ns TEST data...")

# Load TEST trajectory directly
traj = md.load("data/md/raw/trajectory.dcd",
               top="data/molecules/alanine_final.pdb")

# Use center-of-mass
traj_2d = traj.xyz[:, :, :2]  # (frames, atoms, 2)
com_positions = traj_2d.mean(axis=1)  # (frames, 2)
x = com_positions[:, 0]
y = com_positions[:, 1]

# Simple histogram instead of KDE for reliability
plt.figure(figsize=(10, 8))
plt.hist2d(x, y, bins=50, cmap='plasma')
plt.colorbar(label='Count')

plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title('Position Density: Alanine Dipeptide (1 ns TEST)')

# Save
os.makedirs("figures/raw", exist_ok=True)
plt.savefig("figures/raw/fig2b_density_raw_TEST.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/raw/fig2b_density_raw_TEST.png")

# Save 2D data
os.makedirs("data/md/analysis", exist_ok=True)
np.save("data/md/analysis/probability_density_2d_TEST.npy", com_positions)
print("✓ Saved density data")
