#!/usr/bin/env python3
"""
05_rmsd_convergence.py
RMSD vs time for structural stability (Fig S2b).
"""
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

print("Generating RMSD convergence plot...")

# Load trajectory
traj = md.load("data/md/raw/trajectory_100ns.dcd",
               top="data/molecules/alanine_final.pdb")

# Compute RMSD to first frame (backbone atoms)
backbone_atoms = traj.topology.select("backbone")
backbone_traj = traj.atom_slice(backbone_atoms)
rmsd = md.rmsd(backbone_traj, backbone_traj, 0)  # Frame 0 as reference

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Main plot (time in ns)
ax1.plot(traj.time / 1000, rmsd, 'b-', linewidth=1.5)
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('RMSD (nm)')
ax1.set_title('RMSD Convergence: Alanine Dipeptide (100 ns)')
ax1.grid(True, alpha=0.3)

# Horizontal line at convergence threshold
convergence_threshold = 0.15  # nm
ax1.axhline(y=convergence_threshold, color='r', linestyle='--',
            label=f'Convergence threshold ({convergence_threshold} nm)')
ax1.legend()

# Inset: Last 20 ns
last_20ns_idx = int(0.8 * len(rmsd))  # Last 20%
ax2.plot(traj.time[last_20ns_idx:] / 1000, rmsd[last_20ns_idx:], 'g-', linewidth=2)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('RMSD (nm)')
ax2.set_title('Last 20 ns (Stability Check)')
ax2.grid(True, alpha=0.3)

# Calculate and mark convergence time
converged_idx = np.where(rmsd < convergence_threshold)[0]
if len(converged_idx) > 0:
    convergence_time_ns = traj.time[converged_idx[0]] / 1000
    ax1.axvline(x=convergence_time_ns, color='g', linestyle=':',
                label=f'Converged at {convergence_time_ns:.1f} ns')
    ax1.legend()
    print(f"✓ RMSD converged at {convergence_time_ns:.1f} ns")

plt.tight_layout()

# Save
os.makedirs("figures/supplementary", exist_ok=True)
plt.savefig("figures/supplementary/figS2b_rmsd_convergence.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/supplementary/figS2b_rmsd_convergence.png")

# Save data
np.save("data/md/analysis/rmsd_values.npy", rmsd)
np.save("data/md/analysis/rmsd_time.npy", traj.time)
