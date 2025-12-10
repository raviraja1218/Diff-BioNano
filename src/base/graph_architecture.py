"""
Abstract classes for differentiable computational graph components.
"""

from abc import ABC, abstractmethod

class DesignToMaterial(ABC):
    @abstractmethod
    def __call__(self, design_params):
        """Map design parameters (e.g. 200x200) -> material distribution ε(x,y)."""
        raise NotImplementedError

class FDTDSolver(ABC):
    @abstractmethod
    def run(self, material_distribution):
        """Run FDTD and return E(x,y,t)."""
        raise NotImplementedError

class MolecularTrajectory(ABC):
    @abstractmethod
    def run(self):
        """Return molecular positions r_m(t)."""
        raise NotImplementedError

class SignalFunctional(ABC):
    @abstractmethod
    def compute(self, fields, positions):
        """Compute signal S from fields and molecular positions."""
        raise NotImplementedError
