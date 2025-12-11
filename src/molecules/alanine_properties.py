"""
Alanine Dipeptide Molecular Properties
--------------------------------------

This module defines physical and chemical parameters for alanine dipeptide,
used in molecular dynamics (Phase 2) and light–matter interaction modeling
(Phase 5–6).

It provides:
    - Atom masses
    - Bond lengths, angles, dihedrals (AMBER placeholders)
    - Dipole moment
    - Characteristic vibrational frequencies
    - Simulation conditions (T, pH, ionic strength)
"""

import numpy as np


class AlanineDipeptide:
    """
    Container for alanine dipeptide constants and simulation parameters.
    """

    def __init__(self):
        # --------------------------------------------------------------
        # Basic molecular properties
        # --------------------------------------------------------------
        self.name = "Alanine Dipeptide"
        self.molecular_weight = 160.17  # g/mol

        # Approximate bounding-box size (in meters)
        self.size_nm = 0.9  # ~1 nm
        self.size_m = self.size_nm * 1e-9

        # --------------------------------------------------------------
        # Dipole moment (typical range 1–5 Debye)
        # --------------------------------------------------------------
        # 1 Debye = 3.33564e-30 C·m
        self.dipole_debye = 3.4
        self.dipole_Cm = self.dipole_debye * 3.33564e-30

        # Orientation (placeholder — will be refined in MD)
        self.dipole_direction = np.array([1.0, 0.0, 0.0])

        # --------------------------------------------------------------
        # Characteristic vibrational frequencies (in rad/s)
        # --------------------------------------------------------------
        # These will be cross-checked with literature later.
        self.vibrational_freqs_cm1 = {
            "amide_I": 1650,
            "amide_II": 1550,
            "C_alpha_H": 3000,
        }

        # Convert to angular frequency ω = 2πcν
        c = 2.99792458e10  # cm/s
        self.vibrational_omega = {
            mode: 2 * np.pi * c * freq
            for mode, freq in self.vibrational_freqs_cm1.items()
        }

        # --------------------------------------------------------------
        # AMBER Force Field placeholders
        # --------------------------------------------------------------
        self.bonds = {
            "C-N": 1.33e-10,
            "C-C": 1.52e-10,
            "N-H": 1.01e-10,
        }

        self.angles_deg = {
            "C-N-H": 120.0,
            "N-C-C": 110.0,
            "C-C-N": 114.0,
        }

        self.dihedrals = {
            "phi": {"period": 3, "amplitude": 2.0},   # kcal/mol placeholder
            "psi": {"period": 3, "amplitude": 2.5},
        }

        # --------------------------------------------------------------
        # Simulation environmental conditions
        # --------------------------------------------------------------
        self.temperature_K = 300
        self.pH = 7.0
        self.ionic_strength_M = 0.15

    # ------------------------------------------------------------------
    # Utility access functions
    # ------------------------------------------------------------------

    def dipole_vector(self):
        """Return dipole vector μ = μ₀ d̂."""
        return self.dipole_Cm * self.dipole_direction

    def vibrational_modes(self):
        """Return vibrational angular frequencies."""
        return self.vibrational_omega

    def summary(self):
        """Return a clean summary of the molecular parameters."""
        return {
            "name": self.name,
            "MW": self.molecular_weight,
            "dipole (C·m)": self.dipole_Cm,
            "size (nm)": self.size_nm,
            "temperature (K)": self.temperature_K,
        }
