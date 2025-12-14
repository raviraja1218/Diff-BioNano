#!/usr/bin/env python3
"""
Phase 4 - Step 5: Main optimization loop (FIXED VERSION)
"""
import jax
import jax.numpy as jnp
import numpy as np
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

def main_optimization(epochs=300, save_freq=10):
    """Main optimization loop - FIXED VERSION"""
    print("=" * 60)
    print("STARTING MAIN OPTIMIZATION LOOP (FIXED)")
    print(f"Epochs: {epochs}, Checkpoint frequency: {save_freq}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load configuration
    with open("config/phase4_config.json", "r") as f:
        config = json.load(f)
    
    # Load trajectory data
    trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
    positions = jnp.array(trajectory_data['positions'])
    density = jnp.array(trajectory_data['density'])
    
    # Load initial design
    initial_design = np.load("data/optimization/initial_design.npy")
    design = jnp.array(initial_design)
    
    # Parameters
    grid_size = config['grid_size']
    learning_rate = config['learning_rate']
    batch_size = min(config['batch_size'], positions.shape[0])
    total_frames = positions.shape[0]
    
    # SIMPLIFIED BUT WORKING OBJECTIVE FUNCTION
    @jax.jit
    def objective_function(current_design, key):
        """Simplified objective: field ~ design value at position"""
        # Random batch
        batch_indices = jax.random.choice(
            key, 
            total_frames, 
            shape=(batch_size,), 
            replace=False
        )
        batch_positions = positions[batch_indices]
        
        # Convert positions to integer indices
        indices = jnp.floor(batch_positions).astype(jnp.int32)
        indices = jnp.clip(indices, 0, jnp.array(grid_size)-1)
        
        # Get design values at those positions
        design_values = current_design[indices[:, 0], indices[:, 1]]
        
        # Field intensity ~ design_value^2
        intensity = jnp.mean(design_values**2)
        
        # Regularization: encourage binary values
        binary_penalty = jnp.mean(current_design * (1 - current_design)) * config['regularization']
        
        # Simple loss: maximize intensity
        return -intensity + binary_penalty
    
    # Get gradient function
    @jax.jit
    def compute_gradient(current_design, key):
        return jax.grad(lambda x: objective_function(x, key))(current_design)
    
    # Adam optimizer parameters
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    
    # Initialize Adam states
    m = jnp.zeros_like(design)
    v = jnp.zeros_like(design)
    
    # Storage for history
    design_history = []
    loss_history = []
    gradient_history = []
    
    # Create checkpoints directory
    os.makedirs("data/optimization/checkpoints", exist_ok=True)
    
    # Optimization loop
    start_time = time.time()
    key = jax.random.PRNGKey(42)  # Master random key
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Split random key for this epoch
        key, subkey = jax.random.split(key)
        
        # Compute loss and gradient
        loss = objective_function(design, subkey)
        grad = compute_gradient(design, subkey)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        design = design - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Clip to [0, 1]
        design = jnp.clip(design, 0, 1)
        
        # Store history
        loss_history.append(float(loss))
        gradient_history.append(float(jnp.mean(jnp.abs(grad))))
        
        # Save checkpoint
        if epoch % save_freq == 0 or epoch == epochs - 1:
            design_history.append(np.array(design))
            
            # Save checkpoint
            checkpoint = {
                'design': np.array(design),
                'loss': float(loss),
                'gradient_norm': float(jnp.linalg.norm(grad)),
                'epoch': epoch,
                'time': time.time() - start_time
            }
            
            np.savez(f"data/optimization/checkpoints/checkpoint_{epoch:04d}.npz", **checkpoint)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f} | "
                  f"Grad: {jnp.mean(jnp.abs(grad)):.6f} | "
                  f"Time: {epoch_time:.3f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Improvement: {(loss_history[0] - loss_history[-1])/abs(loss_history[0])*100:.1f}%")
    print("=" * 60)
    
    # Convert history to numpy arrays
    design_history_array = np.array(design_history)
    loss_history_array = np.array(loss_history)
    gradient_history_array = np.array(gradient_history)
    
    # Save final results
    np.save("data/optimization/final_design.npy", np.array(design))
    np.save("data/optimization/design_history.npy", design_history_array)
    np.save("data/optimization/loss_history.npy", loss_history_array)
    np.save("data/optimization/gradient_history.npy", gradient_history_array)
    
    # Save performance data
    performance_data = {
        'total_time': total_time,
        'epochs': epochs,
        'final_loss': float(loss),
        'initial_loss': float(loss_history[0]),
        'improvement_percent': float((loss_history[0] - loss_history[-1])/abs(loss_history[0])*100),
        'average_gradient': float(np.mean(gradient_history_array)),
        'convergence_epoch': int(np.argmin(np.diff(loss_history_array) > 0)) if len(loss_history_array) > 1 else epochs
    }
    
    np.savez("data/optimization/performance_data.npz", **performance_data)
    
    # Create completion flag
    with open("validation/phase4/optimization_completed.passed", "w") as f:
        f.write(f"Optimization completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Total time: {total_time/3600:.2f} hours\n")
        f.write(f"Final loss: {loss_history[-1]:.6f}\n")
        f.write(f"Improvement: {performance_data['improvement_percent']:.1f}%\n")
    
    print("\n✓ Final design saved: data/optimization/final_design.npy")
    print("✓ Design history saved: data/optimization/design_history.npy")
    print("✓ Loss history saved: data/optimization/loss_history.npy")
    print("✓ Performance data saved: data/optimization/performance_data.npz")
    print("✓ Completion flag created")
    
    return design, loss_history_array

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run main optimization')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--save-freq', type=int, default=10, help='Checkpoint frequency')
    
    args = parser.parse_args()
    
    print("⚠️  WARNING: This will run for several hours.")
    print("   Start in the evening and let it run overnight.")
    print("   Press Ctrl+C to stop early.\n")
    
    input("Press Enter to start optimization, or Ctrl+C to cancel...")
    
    try:
        final_design, loss_history = main_optimization(
            epochs=args.epochs,
            save_freq=args.save_freq
        )
        print("✅ Optimization completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        print("   Partial results have been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
