#!/usr/bin/env python3
"""
05_rmsd_convergence_FIXED.py
Use TEST trajectory data
"""
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating RMSD convergence plot from 1 ns TEST data...")

# Load TEST trajectory
traj = md.load("data/md/raw/trajectory.dcd",
               top="data/molecules/alanine_final.pdb")

print(f"Using {traj.n_frames} frames")

# Compute RMSD to first frame
backbone_atoms = traj.topology.select("backbone")
if len(backbone_atoms) == 0:
    print("⚠ No backbone atoms found, using all atoms")
    backbone_atoms = traj.topology.select("all")

backbone_traj = traj.atom_slice(backbone_atoms)
rmsd = md.rmsd(backbone_traj, backbone_traj, 0)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(traj.time, rmsd, 'b-', linewidth=2)
ax.set_xlabel('Time (ps)')
ax.set_ylabel('RMSD (nm)')
ax.set_title('RMSD Convergence: Alanine Dipeptide (1 ns TEST)')
ax.grid(True, alpha=0.3)

# Save
os.makedirs("figures/supplementary", exist_ok=True)
plt.savefig("figures/supplementary/figS2b_rmsd_convergence_TEST.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/supplementary/figS2b_rmsd_convergence_TEST.png")

# Save data
os.makedirs("data/md/analysis", exist_ok=True)
np.save("data/md/analysis/rmsd_values_TEST.npy", rmsd)
np.save("data/md/analysis/rmsd_time_TEST.npy", traj.time)
print("✓ Saved RMSD data")
