"""
Material Library for Diff-BioNano
---------------------------------

Provides:
    - Gold permittivity model (Drude-Lorentz)
    - Water refractive index interpolation
    - Frequency/wavelength conversion utilities
    - ε(ω) interpolation for FDTD

This module is used by:
    Phase 3 (FDTD)
    Phase 5 (Spectral analysis)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# -------------------------------------------------------------------------
# 1. Load experimental material data
# -------------------------------------------------------------------------

def load_gold_permittivity(csv_path="data/materials/gold_permittivity.csv"):
    """
    Loads gold permittivity dataset.
    CSV must contain:
        wavelength_nm, real_eps, imag_eps
    """
    df = pd.read_csv(csv_path)
    return df


def load_water_index(csv_path="data/materials/water_refractive_index.csv"):
    """
    Loads refractive index of water dataset.
    Columns:
        wavelength_nm, n
    """
    df = pd.read_csv(csv_path)
    return df


# -------------------------------------------------------------------------
# 2. Drude-Lorentz model for Au
# -------------------------------------------------------------------------

def drude_lorentz_epsilon(omega, wp, gamma, eps_inf):
    """
    Drude model:
        ε(ω) = ε_inf - wp^2 / (ω^2 + i γ ω)
    You will refine with Lorentz terms if needed.
    """
    return eps_inf - (wp**2) / (omega**2 + 1j * gamma * omega)


# -------------------------------------------------------------------------
# 3. Interpolation functions
# -------------------------------------------------------------------------

def interpolate_gold(df):
    """
    Returns λ → ε interpolation functions for gold.
    """
    wl = df["wavelength_nm"].values
    eps_r = df["real_eps"].values
    eps_i = df["imag_eps"].values

    real_interp = interp1d(wl, eps_r, kind="cubic", fill_value="extrapolate")
    imag_interp = interp1d(wl, eps_i, kind="cubic", fill_value="extrapolate")

    def epsilon_lambda(lambda_nm):
        return real_interp(lambda_nm) + 1j * imag_interp(lambda_nm)

    return epsilon_lambda


def interpolate_water(df):
    wl = df["wavelength_nm"].values
    n = df["n"].values

    n_interp = interp1d(wl, n, kind="cubic", fill_value="extrapolate")

    def n_lambda(lambda_nm):
        return n_interp(lambda_nm)

    return n_lambda


# -------------------------------------------------------------------------
# 4. Utility conversions
# -------------------------------------------------------------------------

def lambda_to_omega(lambda_nm):
    c = 2.99792458e8
    lambda_m = lambda_nm * 1e-9
    return 2 * np.pi * c / lambda_m


def omega_to_lambda(omega):
    c = 2.99792458e8
    lambda_m = 2 * np.pi * c / omega
    return lambda_m * 1e9  # nm


# -------------------------------------------------------------------------
# 5. Unified material interface
# -------------------------------------------------------------------------

class MaterialLibrary:
    """
    Unified interface for:
        ε_gold(λ)
        n_water(λ)
        Conversion between ω and λ
    """

    def __init__(self):
        self.gold_df = load_gold_permittivity()
        self.water_df = load_water_index()

        self.eps_gold = interpolate_gold(self.gold_df)
        self.n_water = interpolate_water(self.water_df)

    def epsilon_gold(self, lambda_nm):
        return self.eps_gold(lambda_nm)

    def refractive_index_water(self, lambda_nm):
        return self.n_water(lambda_nm)

    def epsilon_water(self, lambda_nm):
        """Return ε = n^2."""
        n = self.refractive_index_water(lambda_nm)
        return n**2
