"""
Adjoint Maxwell Solver — Theoretical Scaffold
---------------------------------------------

This module provides the abstract structure for implementing
adjoint Maxwell equations in the differentiable FDTD pipeline.

It does NOT implement the numerical solver (Phase 3 will do that).
Instead, it defines the API, tensor flows, and gradient hooks.

Forward: 
    E, H → objective S

Backward (Adjoint):
    dS/dE, dS/dH → λ_E, λ_H → dS/d(material parameters)
"""

from abc import ABC, abstractmethod


class AdjointSolver(ABC):
    """
    Abstract base class for adjoint Maxwell solver.
    """

    @abstractmethod
    def initialize(self, grid_shape, dt, material_params):
        """
        Prepare adjoint field arrays, PML profiles, and buffers.

        Args:
            grid_shape: (Nx, Ny) or (Nx, Ny, Nz)
            dt: time step (float)
            material_params: dictionary containing ε(x,y) or ε(ω)
        """
        raise NotImplementedError

    @abstractmethod
    def run_backward(self, dS_dE, dS_dH):
        """
        Integrate adjoint equations backward in time.

        Args:
            dS_dE: ∂S/∂E(x,y,t)   — source term from optical signal
            dS_dH: ∂S/∂H(x,y,t)

        Returns:
            lambda_E, lambda_H: adjoint fields for all timesteps
        """
        raise NotImplementedError

    @abstractmethod
    def compute_design_gradient(self, lambda_E, E_forward):
        """
        Compute derivative of the objective w.r.t design parameters.

        Implements:
            dS/dρ = - ∫ λ_E ⋅ ( ∂ε/∂ρ ) ⋅ E  dt

        Args:
            lambda_E: adjoint electric field
            E_forward: forward electric field from FDTD

        Returns:
            grad: ndarray matching design_params shape (H, W)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# Optional: helper functions (will be useful later)
# ---------------------------------------------------------------------

def fdtd_time_reverse(arr):
    """
    Utility for reversing time dimension.
    E.g. E(t) → E(T - t)
    """
    return arr[::-1].copy()


def contract_fields(lambda_E, E_forward, deps_drho):
    """
    Compute contraction: integral(lambda_E * deps/drho * E dt)
    This is the core adjoint gradient.
    """
    return (lambda_E * deps_drho * E_forward).sum()
