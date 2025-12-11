"""
Physical scales and dimensionless parameters for Diff-BioNano.

These scales define the natural units for the coupled
molecule–plasmon electromagnetic system.

You will refine these values after completing the full
Hamiltonian analysis, but these placeholders allow
Phase 1 validation + scaling checks to proceed.
"""

import numpy as np

# -------------------------------------------------------------------------
# 1. Core characteristic scales (chosen based on plasmonic + molecular physics)
# -------------------------------------------------------------------------

PHYSICAL_SCALES = {
    # Characteristic plasmonic feature size (10 nm typical)
    "L0_m": 10e-9,

    # Characteristic optical/molecular timescale (~1 femtosecond)
    "T0_s": 1e-15,

    # Characteristic energy scale (1 eV)
    "E0_J": 1.602176634e-19,

    # Reference field amplitude (you will refine later)
    "E0_field": 1.0,

    # Molecular dipole magnitude (typical 1–5 Debye)
    "mu0_Cm": 3.0e-30,
}

# -------------------------------------------------------------------------
# 2. Derived dimensionless groups
# -------------------------------------------------------------------------

def derived_dimensionless():
    c = 2.99792458e8       # Speed of light (m/s)
    L0 = PHYSICAL_SCALES["L0_m"]
    T0 = PHYSICAL_SCALES["T0_s"]

    return {
        # Courant / EM scaling
        "cT0_over_L0": c * T0 / L0,

        # Domain-to-wavelength ratio — filled later in Phase 2/3
        "lambda_over_L0": None,

        # Time step over characteristic time (will be set after CFL condition)
        "dt_over_T0": None,

        # Dimensionless coupling strength placeholder
        "g_dimensionless": None,
    }

# -------------------------------------------------------------------------
# 3. Debug print helper
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Derived Dimensionless Parameters ===")
    for k, v in derived_dimensionless().items():
        print(f"{k}: {v}")
