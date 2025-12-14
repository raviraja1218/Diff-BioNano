#!/usr/bin/env python3
"""
Phase 4 - Step 2: Define differentiable objective function
"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def create_differentiable_objective():
    """Create the differentiable loss function for optimization"""
    print("Creating differentiable objective function...")
    
    # Load configuration
    with open("config/phase4_config.json", "r") as f:
        config = json.load(f)
    
    # Load trajectory
    trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
    positions = jnp.array(trajectory_data['positions'])  # (frames, 2)
    
    # Parameters from config
    grid_size = config['grid_size']
    dx = config['dx']
    wavelength = config['wavelength']
    
    # Simple field calculation (placeholder - will be replaced with real FDTD)
    # This function simulates field intensity at a position given a design
    @jit
    def field_at_position(design, position):
        """Calculate field intensity at a position for a given design"""
        # Convert position to grid indices
        i, j = jnp.floor(position).astype(jnp.int32)
        i = jnp.clip(i, 0, grid_size[0]-1)
        j = jnp.clip(j, 0, grid_size[1]-1)
        
        # Simple field model: intensity ~ design value at that pixel
        # Gold (design=1) gives high field, water (design=0) gives low field
        field_strength = design[i, j] * 10.0  # Arbitrary scaling
        
        return field_strength
    
    # Vectorize over positions
    batched_field = vmap(field_at_position, in_axes=(None, 0))
    
    @jit
    def objective_function(design):
        """Objective: Maximize average field intensity at molecule positions"""
        # Regularization: encourage binary designs (0 or 1)
        binary_loss = jnp.mean(design * (1 - design)) * config['regularization']
        
        # Field intensity at all positions
        fields = batched_field(design, positions)
        
        # Average intensity (we want to maximize this)
        intensity = jnp.mean(fields**2)
        
        # Loss = -intensity + regularization (negative because we maximize)
        loss = -intensity + binary_loss
        
        return loss
    
    # Create gradient function
    gradient_function = jit(grad(objective_function))
    
    # Test with a random design
    test_design = jnp.ones((grid_size[0], grid_size[1])) * 0.5
    test_loss = objective_function(test_design)
    test_grad = gradient_function(test_design)
    
    print(f"✓ Objective function created")
    print(f"✓ Test loss: {test_loss:.4f}")
    print(f"✓ Gradient shape: {test_grad.shape}")
    print(f"✓ Gradient mean: {jnp.mean(test_grad):.6f}")
    
    # Save the functions for later use
    save_data = {
        'objective': objective_function,
        'gradient': gradient_function,
        'config': config
    }
    
    # Can't save JAX functions directly, but we can save metadata
    np.savez("data/optimization/objective_functions.npz", 
             config=np.array([config['learning_rate'], config['regularization']]))
    
    return objective_function, gradient_function, config

if __name__ == "__main__":
    obj_func, grad_func, config = create_differentiable_objective()
    print("✅ Step 2 complete: Objective function defined")
