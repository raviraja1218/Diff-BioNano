#!/usr/bin/env python3
"""
Phase 4 - Step 3: Create initial random design
"""
import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

def create_initial_designs():
    """Create initial designs for optimization"""
    print("Creating initial designs...")
    
    # Load configuration
    with open("config/phase4_config.json", "r") as f:
        config = json.load(f)
    
    grid_size = config['grid_size']
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    
    # Create different initial designs
    designs = {}
    
    # 1. Random design (our starting point)
    key, subkey = jax.random.split(key)
    random_design = jax.random.uniform(subkey, grid_size)
    designs['random'] = random_design
    
    # 2. Nanodisk baseline (for comparison)
    nanodisk_design = jnp.zeros(grid_size)
    center = jnp.array(grid_size) // 2
    radius = 30  # pixels
    
    # Create circular nanodisk
    x, y = jnp.meshgrid(jnp.arange(grid_size[0]), jnp.arange(grid_size[1]), indexing='ij')
    distance = jnp.sqrt((x - center[0])**2 + (y - center[1])**2)
    nanodisk_design = jnp.where(distance < radius, 1.0, 0.0)
    designs['nanodisk'] = nanodisk_design
    
    # 3. Uniform design
    uniform_design = jnp.ones(grid_size) * 0.5
    designs['uniform'] = uniform_design
    
    # Save designs
    for name, design in designs.items():
        np.save(f"data/optimization/{name}_design.npy", np.array(design))
        print(f"✓ Saved {name} design: {design.shape}")
    
    # Save our starting design (random)
    np.save("data/optimization/initial_design.npy", np.array(random_design))
    
    # Create visualization of initial designs
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (name, design) in enumerate(designs.items()):
        ax = axes[idx]
        im = ax.imshow(design.T, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'{name.capitalize()} Design')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('figures/analysis/initial_designs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created visualization: figures/analysis/initial_designs.png")
    
    return designs

if __name__ == "__main__":
    designs = create_initial_designs()
    print("✅ Step 3 complete: Initial designs created")
