#!/usr/bin/env python3
"""
07_md_statistics_table.py
Create LaTeX table for Methods section.
"""
import numpy as np
import pandas as pd
import json
import os

print("Creating MD statistics table...")

# Load metadata
with open("data/md/processed/metadata.json", "r") as f:
    meta = json.load(f)

# Load state data
state_data = pd.read_csv("data/md/raw/state_100ns.csv")

# Calculate statistics
temp_mean = state_data['Temperature (K)'].mean()
temp_std = state_data['Temperature (K)'].std()
energy_mean = state_data['Potential Energy (kJ/mole)'].mean()
energy_std = state_data['Potential Energy (kJ/mole)'].std()
speed_mean = state_data['Speed (ns/day)'].mean()

# Load RMSD for convergence time (from previous script)
rmsd = np.load("data/md/analysis/rmsd_values.npy")
time_ps = np.load("data/md/analysis/rmsd_time.npy")
convergence_threshold = 0.15  # nm
converged_idx = np.where(rmsd < convergence_threshold)[0]
convergence_time_ns = time_ps[converged_idx[0]]/1000 if len(converged_idx) > 0 else 0

# Create table
table_data = {
    "Parameter": [
        "Simulation time",
        "Number of atoms",
        "Time step",
        "Temperature (mean ± std)",
        "Potential energy (mean ± std)",
        "Performance (mean)",
        "RMSD convergence time",
        "Trajectory frames",
        "Frame interval",
        "Total steps"
    ],
    "Value": [
        f"{meta['time_ps']/1000:.0f} ns",
        f"{meta['atoms']}",
        "2 fs",
        f"{temp_mean:.1f} ± {temp_std:.1f} K",
        f"{energy_mean:.1f} ± {energy_std:.1f} kJ/mol",
        f"{speed_mean:.0f} ns/day",
        f"{convergence_time_ns:.1f} ns",
        f"{meta['frames']}",
        "10 ps",
        f"{meta['frames'] * 5000:,}"  # 5000 steps per frame
    ],
    "Description": [
        "Total production run",
        "ACE-ALA-NME peptide",
        "Standard for biomolecular MD",
        "NVT ensemble at 300 K target",
        "AMBER99SB-ILDN force field",
        "On NVIDIA RTX 4050 GPU",
        f"Time to reach <{convergence_threshold} nm RMSD",
        "Saved to DCD file",
        "Between saved frames",
        "2 fs per step × total time"
    ]
}

# Save as CSV and LaTeX
df = pd.DataFrame(table_data)
os.makedirs("tables", exist_ok=True)
df.to_csv("tables/md_statistics.csv", index=False)

latex_table = df.to_latex(index=False, caption="Molecular Dynamics Simulation Parameters",
                          label="tab:md_params")
with open("tables/md_statistics.tex", "w") as f:
    f.write(latex_table)

print("✓ Saved: tables/md_statistics.csv")
print("✓ Saved: tables/md_statistics.tex (for paper)")
