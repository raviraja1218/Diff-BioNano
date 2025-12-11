# Literature Comparison — Phase 1 Validation

This document summarizes the comparison of our theoretical formulation with
established results from the literature.

## 1. Plasmonic Nanostructure Resonances

Reference: Johnson & Christy (1972), Phys. Rev. B  
- Gold nanodisk (100 nm):
  - Reported resonance: ~600–650 nm
  - Our model target: 600 ± 20 nm

## 2. Molecular Dipole Interaction

Reference: Andrews & Bradshaw, *J. Chem. Phys.*  
- Dipole-field interaction scales as:
  \[
  \Delta S \propto |\mu|^2
  \]
- We test this in weak coupling limit.

## 3. Alanine Dipeptide Dihedral Distribution

Reference: Shaw et al., *Science* (2010)  
- φ/ψ distribution peak positions:
  - α-helix region
  - β-sheet region

## 4. Material Optical Constants

Reference: Johnson & Christy  
- Gold ε(λ) real/imag parts used to validate:
  - cubic interpolation
  - Drude-Lorentz model

## 5. Yee Grid Dispersion Comparison

Reference: Taflove, *Computational Electrodynamics*  
- EM dispersion relation:
  \[
  \omega(k) = \frac{2}{\Delta t} \sin^{-1} \left(
      c \Delta t 
      \sqrt{ 
          \frac{
            \sin^2(k_x \Delta x / 2)
            +
            \sin^2(k_y \Delta y / 2)
          }{
            \Delta x^2
          }
      }
  \right)
  \]
- Used to validate FDTD forward/adjoint discretization.
