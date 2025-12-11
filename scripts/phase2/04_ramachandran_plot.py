#!/usr/bin/env python3
"""
04_ramachandran_plot.py
Generate φ/ψ dihedral angle plot (Fig S2a).
"""
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating Ramachandran plot...")

# Load trajectory
traj = md.load("data/md/raw/trajectory_100ns.dcd",
               top="data/molecules/alanine_final.pdb")

# Compute dihedral angles (in radians, then convert to degrees)
phi_indices, phi_angles = md.compute_phi(traj)
psi_indices, psi_angles = md.compute_psi(traj)

phi_deg = phi_angles.reshape(-1) * 180 / np.pi
psi_deg = psi_angles.reshape(-1) * 180 / np.pi

# Create plot
plt.figure(figsize=(10, 8))
hb = plt.hexbin(phi_deg, psi_deg, gridsize=50, cmap='viridis', bins='log')
plt.colorbar(hb, label='Count (log scale)')

# Mark characteristic regions
plt.axvline(x=-60, color='red', linestyle='--', alpha=0.5, label='α-helix (φ ≈ -60°)')
plt.axhline(y=-60, color='red', linestyle='--', alpha=0.5)
plt.axvline(x=-120, color='blue', linestyle='--', alpha=0.5, label='β-sheet (φ ≈ -120°)')
plt.axhline(y=120, color='blue', linestyle='--', alpha=0.5)

plt.xlabel('φ (degrees)')
plt.ylabel('ψ (degrees)')
plt.title('Ramachandran Plot: Alanine Dipeptide (100 ns)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save
os.makedirs("figures/supplementary", exist_ok=True)
plt.savefig("figures/supplementary/figS2a_ramachandran.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/supplementary/figS2a_ramachandran.png")

# Save data
os.makedirs("data/md/analysis", exist_ok=True)
np.save("data/md/analysis/phi_angles.npy", phi_deg)
np.save("data/md/analysis/psi_angles.npy", psi_deg)
print(f"✓ Saved dihedral data")
