"""
Computational Graph Architecture for Diff-BioNano
-------------------------------------------------

This file defines the abstract classes that make up the full
differentiable simulation pipeline:

    design → material → FDTD → molecular trajectory → signal functional

Each module must be differentiable (or provide adjoint hooks)
and must follow a strict interface so the optimization pipeline
(Phase 4) can call everything consistently.

You will implement concrete subclasses in later phases.
"""

from abc import ABC, abstractmethod


# -------------------------------------------------------------------------
# 1. Design → Material Distribution
# -------------------------------------------------------------------------
class DesignToMaterial(ABC):
    """
    Converts a design parameter array (e.g. 200×200 float tensor)
    into a material distribution ε(x,y).
    """

    @abstractmethod
    def __call__(self, design_params):
        """
        Args:
            design_params: ndarray or JAX array of shape (H, W)
        Returns:
            material_dist: dict or tensor containing ε(x,y) values
        """
        raise NotImplementedError


# -------------------------------------------------------------------------
# 2. Electromagnetic FDTD Solver
# -------------------------------------------------------------------------
class FDTDSolver(ABC):
    """
    Runs a 2D/3D FDTD simulation on a Yee grid.
    """

    @abstractmethod
    def run(self, material_distribution):
        """
        Args:
            material_distribution: ε(x,y) or tensor of permittivity values
        Returns:
            fields: dictionary containing E(x,y,t) and H(x,y,t)
        """
        raise NotImplementedError

    @abstractmethod
    def get_time_step(self):
        """Return dt determined by CFL condition."""
        raise NotImplementedError


# -------------------------------------------------------------------------
# 3. Molecular Trajectory Generator (from MD)
# -------------------------------------------------------------------------
class MolecularTrajectory(ABC):
    """
    Produces the molecular positions r_m(t) from MD or analytical model.
    """

    @abstractmethod
    def run(self):
        """
        Returns:
            positions: array of shape (T, N_molecules, 3)
        """
        raise NotImplementedError


# -------------------------------------------------------------------------
# 4. EM Field → Molecule Interpolation
# -------------------------------------------------------------------------
class FieldSampler(ABC):
    """
    Samples the EM field at the molecular positions.
    """

    @abstractmethod
    def sample(self, fields, positions):
        """
        Args:
            fields: dict of E(x,y,t), H(x,y,t)
            positions: r_m(t) from trajectory
        Returns:
            sampled_fields: E(r_m(t)), shape (T, N_molecules, 3)
        """
        raise NotImplementedError


# -------------------------------------------------------------------------
# 5. Signal Functional
# -------------------------------------------------------------------------
class SignalFunctional(ABC):
    """
    Converts sampled fields into a measurable scalar signal S.

    Example: S = ∫ |E(r_m(t))|^2 dt
    """

    @abstractmethod
    def compute(self, sampled_fields):
        """
        Args:
            sampled_fields: array of E(r_m(t))
        Returns:
            S: scalar loss or signal value
        """
        raise NotImplementedError


# -------------------------------------------------------------------------
# 6. Full Pipeline Wrapper
# -------------------------------------------------------------------------
class DifferentiablePipeline(ABC):
    """
    Abstract wrapper for the entire pipeline.
    Concrete implementation will plug all modules together.
    """

    @abstractmethod
    def forward(self, design_params):
        """
        Full forward pass:
            design → material → FDTD → sample → signal
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output):
        """
        Full backward/adjoint pass:
            signal → adjoint → gradients wrt design_params
        """
        raise NotImplementedError
