#!/usr/bin/env python3
"""
07_md_statistics_table_FIXED.py
Create statistics using TEST data
"""
import numpy as np
import pandas as pd
import os

print("Creating MD statistics table from TEST data...")

# Load TEST state data
state_file = "data/md/raw/state.csv"  # TEST file
state_data = pd.read_csv(state_file)

# Calculate statistics from TEST
temp_mean = state_data['Temperature (K)'].mean()
temp_std = state_data['Temperature (K)'].std()
energy_mean = state_data['Potential Energy (kJ/mole)'].mean()
energy_std = state_data['Potential Energy (kJ/mole)'].std()
speed_mean = state_data['Speed (ns/day)'].mean()

# Create table with TEST note
table_data = {
    "Parameter": [
        "Simulation time (TEST)",
        "Number of atoms",
        "Time step",
        "Temperature (mean ± std)",
        "Potential energy (mean ± std)",
        "Performance (mean)",
        "Trajectory frames",
        "Frame interval",
        "Note"
    ],
    "Value": [
        "1 ns",
        "22",
        "2 fs",
        f"{temp_mean:.1f} ± {temp_std:.1f} K",
        f"{energy_mean:.1f} ± {energy_std:.1f} kJ/mol",
        f"{speed_mean:.0f} ns/day",
        "100",
        "10 ps",
        "Using 1 ns test data for validation"
    ],
    "Description": [
        "Test run for validation",
        "ACE-ALA-NME peptide",
        "Standard for biomolecular MD",
        "NVT ensemble at 300 K target",
        "AMBER99SB-ILDN force field",
        "On NVIDIA RTX 4050 GPU",
        "Saved to DCD file",
        "Between saved frames",
        "100 ns production file corrupted"
    ]
}

# Save
df = pd.DataFrame(table_data)
os.makedirs("tables", exist_ok=True)
df.to_csv("tables/md_statistics_TEST.csv", index=False)

latex_table = df.to_latex(index=False, caption="Molecular Dynamics Simulation Parameters (TEST)",
                          label="tab:md_params_test")
with open("tables/md_statistics_TEST.tex", "w") as f:
    f.write(latex_table)

print("✓ Saved: tables/md_statistics_TEST.csv")
print("✓ Saved: tables/md_statistics_TEST.tex")
