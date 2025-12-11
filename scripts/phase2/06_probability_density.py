#!/usr/bin/env python3
"""
06_probability_density.py
2D position probability heatmap (Fig 2b raw).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

print("Generating probability density map...")

# Load 2D trajectory
traj_2d = np.load("data/md/processed/trajectory_2d.npy")  # (frames, atoms, 2)

# Use center-of-mass (average over all atoms)
com_positions = traj_2d.mean(axis=1)  # Shape: (frames, 2)
x = com_positions[:, 0]
y = com_positions[:, 1]

# Kernel Density Estimation
positions = np.vstack([x, y])
kde = gaussian_kde(positions)

# Create grid
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi.reshape(xi.shape)  # Shape: (100, 100)

# Create plot
plt.figure(figsize=(10, 8))
plt.contourf(xi, yi, zi, levels=50, cmap='plasma', alpha=0.8)
plt.colorbar(label='Probability Density')

# Overlay trajectory samples
plt.scatter(x[::100], y[::100], c='white', s=1, alpha=0.3, label='Trajectory samples')

plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title('Probability Density: Alanine Dipeptide Center-of-Mass (100 ns)')
plt.legend()

# Save
os.makedirs("figures/raw", exist_ok=True)
plt.savefig("figures/raw/fig2b_density_raw.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/raw/fig2b_density_raw.png")

# Save density data for Phase 4 (sensor optimization)
os.makedirs("data/md/analysis", exist_ok=True)
np.save("data/md/analysis/probability_density_2d.npy", zi)
np.save("data/md/analysis/density_grid_x.npy", xi)
np.save("data/md/analysis/density_grid_y.npy", yi)
print(f"✓ Saved density grid data")
