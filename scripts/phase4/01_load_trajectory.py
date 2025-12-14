#!/usr/bin/env python3
"""
Phase 4 - Step 1: Load molecular dynamics trajectory
"""
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_phase2_data():
    """Load trajectory from Phase 2"""
    print("Loading Phase 2 trajectory data...")
    
    # Load 2D trajectory (from Phase 2)
    trajectory_path = "data/md/processed/trajectory_2d_TEST.npy"
    density_path = "data/md/analysis/probability_density_2d_TEST.npy"
    
    try:
        trajectory_2d = np.load(trajectory_path)
        probability_density = np.load(density_path)
        
        print(f"✓ Loaded trajectory: {trajectory_2d.shape}")
        print(f"✓ Loaded density: {probability_density.shape}")
        
        # Process for FDTD: get center of mass positions
        # trajectory_2d shape: (frames, atoms, 2)
        # Take average over atoms for molecule center
        molecule_positions = np.mean(trajectory_2d, axis=1)  # (frames, 2)
        
        # Convert from nanometers to grid coordinates
        dx = 1e-9  # 1 nm per pixel
        molecule_grid_positions = molecule_positions / dx  # Convert nm to grid units
        
        # Clip to grid bounds
        molecule_grid_positions = np.clip(molecule_grid_positions, 0, 199)
        
        # Save processed data
        save_data = {
            'positions': molecule_grid_positions.astype(np.float32),
            'density': probability_density.astype(np.float32),
            'metadata': {
                'frames': trajectory_2d.shape[0],
                'atoms': trajectory_2d.shape[1],
                'grid_size': [200, 200]
            }
        }
        
        np.savez("data/optimization/trajectory_loaded.npz", **save_data)
        print("✓ Saved processed trajectory to data/optimization/trajectory_loaded.npz")
        
        return molecule_grid_positions, probability_density
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run Phase 2 first to generate trajectory data.")
        sys.exit(1)

if __name__ == "__main__":
    load_phase2_data()
    print("✅ Step 1 complete: Trajectory loaded successfully")
