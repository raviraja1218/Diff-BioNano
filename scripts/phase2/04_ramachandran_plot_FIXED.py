#!/usr/bin/env python3
"""
04_ramachandran_plot_FIXED.py
Use TEST trajectory data
"""
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating Ramachandran plot from 1 ns TEST data...")

# Load TEST trajectory
traj = md.load("data/md/raw/trajectory.dcd",  # 1 ns file
               top="data/molecules/alanine_final.pdb")

print(f"Using {traj.n_frames} frames from test trajectory")

# Compute dihedral angles
phi_indices, phi_angles = md.compute_phi(traj)
psi_indices, psi_angles = md.compute_psi(traj)

phi_deg = phi_angles.reshape(-1) * 180 / np.pi
psi_deg = psi_angles.reshape(-1) * 180 / np.pi

# Create plot
plt.figure(figsize=(10, 8))

# Use scatter instead of hexbin to avoid colorbar issues
plt.scatter(phi_deg, psi_deg, s=1, alpha=0.5, c='blue', label='Conformations')

# Mark characteristic regions
plt.axvline(x=-60, color='red', linestyle='--', alpha=0.5, label='α-helix (φ ≈ -60°)')
plt.axhline(y=-60, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=-120, color='blue', linestyle='--', alpha=0.5, label='β-sheet (φ ≈ -120°)')
plt.axhline(y=120, color='blue', linestyle='--', alpha=0.5)

plt.xlabel('φ (degrees)')
plt.ylabel('ψ (degrees)')
plt.title('Ramachandran Plot: Alanine Dipeptide (1 ns TEST)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save
os.makedirs("figures/supplementary", exist_ok=True)
plt.savefig("figures/supplementary/figS2a_ramachandran_TEST.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/supplementary/figS2a_ramachandran_TEST.png")

# Save data
os.makedirs("data/md/analysis", exist_ok=True)
np.save("data/md/analysis/phi_angles_TEST.npy", phi_deg)
np.save("data/md/analysis/psi_angles_TEST.npy", psi_deg)
print("✓ Saved dihedral data")
