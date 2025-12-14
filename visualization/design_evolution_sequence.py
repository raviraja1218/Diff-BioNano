#!/usr/bin/env python3
"""
Generate Figure 2a: Design evolution sequence
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_design_evolution_figure():
    """Create 6-panel figure showing design evolution"""
    print("Generating Figure 2a: Design evolution sequence...")
    
    # Load design history
    design_history = np.load("data/optimization/design_history.npy")
    loss_history = np.load("data/optimization/loss_history.npy")
    
    # Select key epochs to show
    total_checkpoints = design_history.shape[0]
    epoch_indices = [0, 1, 2, 4, 8, total_checkpoints-1]  # Key evolution points
    epoch_numbers = [0, 10, 20, 40, 80, 290]  # Approximate epoch numbers
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each design
    for idx, (epoch_idx, epoch_num) in enumerate(zip(epoch_indices, epoch_numbers)):
        ax = axes[idx]
        design = design_history[epoch_idx]
        
        # Plot design
        im = ax.imshow(design.T, cmap='gray_r', vmin=0, vmax=1,
                      extent=[0, 200, 0, 200])
        
        # Add epoch info
        loss = loss_history[epoch_num] if epoch_num < len(loss_history) else loss_history[-1]
        ax.set_title(f'Epoch {epoch_num}\nLoss: {loss:.3f}', fontsize=10)
        ax.set_xlabel('x (nm)', fontsize=9)
        ax.set_ylabel('y (nm)', fontsize=9)
        
        # Add colorbar to first plot only
        if idx == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Design parameter\n(0=water, 1=gold)', fontsize=8)
    
    plt.suptitle('Design Evolution During Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/results/fig2a_design_evolution.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 2a saved: {output_path}")
    
    # Also create animation for supplementary
    create_evolution_animation(design_history, loss_history)
    
    return True

def create_evolution_animation(design_history, loss_history):
    """Create animation of design evolution for supplementary"""
    import matplotlib.animation as animation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # First frame
    im = ax1.imshow(design_history[0].T, cmap='gray_r', vmin=0, vmax=1,
                   animated=True, extent=[0, 200, 0, 200])
    ax1.set_title('Design Evolution')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im, ax=ax1, label='Design parameter')
    
    # Loss plot
    loss_line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_title('Loss Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(design_history)*10)
    ax2.set_ylim(np.min(loss_history)*1.1, np.max(loss_history)*0.9)
    
    def update(frame):
        """Update animation frame"""
        im.set_array(design_history[frame].T)
        ax1.set_title(f'Design Evolution - Epoch {frame*10}')
        
        # Update loss plot
        epochs = list(range(frame+1))
        losses = loss_history[:frame+1]
        loss_line.set_data(epochs, losses)
        
        return im, loss_line
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(design_history),
                                  interval=200, blit=False)
    
    # Save animation
    output_path = "figures/supplementary/design_evolution_animation.gif"
    anim.save(output_path, writer='pillow', fps=5)
    plt.close()
    
    print(f"✓ Animation saved: {output_path}")
    
    return True

if __name__ == "__main__":
    success = create_design_evolution_figure()
    if success:
        print("✅ Figure 2a generation complete")
    else:
        print("❌ Figure 2a generation failed")
        sys.exit(1)
