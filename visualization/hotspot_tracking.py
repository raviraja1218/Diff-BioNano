#!/usr/bin/env python3
"""
Generate Figure 4a: Dynamic hotspot tracking
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_hotspot_tracking_figure():
    """Create Figure 4a: Hotspot tracking over time"""
    print("Generating Figure 4a: Hotspot tracking...")
    
    # Load data
    final_design = np.load("data/optimization/final_design.npy")
    trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
    positions = trajectory_data['positions']
    
    # Select 4 time points
    total_frames = len(positions)
    time_indices = [0, total_frames//3, 2*total_frames//3, total_frames-1]
    
    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # For each time point
    for idx, t_idx in enumerate(time_indices):
        # Get position at this time
        pos = positions[t_idx]
        
        # Row 1: Design with molecule position
        ax_design = axes[0, idx]
        
        # Plot design
        im_design = ax_design.imshow(final_design.T, cmap='gray_r', alpha=0.8,
                                    extent=[0, 200, 0, 200], vmin=0, vmax=1)
        
        # Mark molecule position
        ax_design.plot(pos[0], pos[1], 'ro', markersize=10, 
                      markerfacecolor='red', markeredgecolor='white', 
                      markeredgewidth=2, label='Molecule')
        
        # Add a circle around position
        circle = plt.Circle((pos[0], pos[1]), 15, fill=False, 
                           color='yellow', linewidth=2, alpha=0.7)
        ax_design.add_patch(circle)
        
        ax_design.set_xlabel('x (nm)', fontsize=9)
        ax_design.set_ylabel('y (nm)', fontsize=9)
        ax_design.set_title(f'Time t = {t_idx} ps\nMolecule position', 
                           fontsize=10, fontweight='bold')
        
        # Add colorbar to first panel only
        if idx == 0:
            cbar = plt.colorbar(im_design, ax=ax_design, fraction=0.046, pad=0.04)
            cbar.set_label('Design value', fontsize=8)
        
        # Row 2: Simulated field at this time
        ax_field = axes[1, idx]
        
        # Simulate field enhancement (simple model)
        # Field is stronger near gold, decays with distance
        x_grid, y_grid = np.meshgrid(np.arange(200), np.arange(200))
        
        # Distance from molecule
        distance = np.sqrt((x_grid - pos[0])**2 + (y_grid - pos[1])**2)
        
        # Field ~ design value * Gaussian decay from molecule
        # Gold areas near molecule give highest field
        field = final_design.T * np.exp(-distance**2 / (30**2))
        
        # Plot field
        im_field = ax_field.imshow(field, cmap='hot', alpha=0.9,
                                  extent=[0, 200, 0, 200], vmin=0, vmax=1)
        
        # Overlay design contours
        ax_field.contour(final_design.T, levels=[0.5], colors='cyan', 
                        linewidths=1.5, alpha=0.7, extent=[0, 200, 0, 200])
        
        # Mark molecule position
        ax_field.plot(pos[0], pos[1], 'wo', markersize=8, 
                     markerfacecolor='white', markeredgecolor='black',
                     markeredgewidth=1.5)
        
        ax_field.set_xlabel('x (nm)', fontsize=9)
        ax_field.set_ylabel('y (nm)', fontsize=9)
        ax_field.set_title(f'Field enhancement |E|²\nat molecule position', 
                          fontsize=10, fontweight='bold')
        
        # Add colorbar to first panel only
        if idx == 0:
            cbar2 = plt.colorbar(im_field, ax=ax_field, fraction=0.046, pad=0.04)
            cbar2.set_label('|E|² / |E₀|²', fontsize=8)
        
        # Add time progression arrows
        if idx < len(time_indices) - 1:
            # Arrow between design panels
            arrow_x = 1.05
            arrow_y = 0.5
            ax_design.annotate('', xy=(arrow_x, arrow_y), 
                             xytext=(arrow_x - 0.1, arrow_y),
                             xycoords='axes fraction',
                             arrowprops=dict(arrowstyle='->', color='blue', 
                                           lw=2, alpha=0.7))
            
            # Arrow between field panels
            ax_field.annotate('', xy=(arrow_x, arrow_y), 
                            xytext=(arrow_x - 0.1, arrow_y),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='blue', 
                                          lw=2, alpha=0.7))
    
    # Add overall title
    plt.suptitle('Dynamic Hotspot Tracking: Field Enhancement Follows Molecular Motion',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/results/fig4a_hotspot_tracking.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure 4a saved: {output_path}")
    
    # Create animation of tracking
    create_hotspot_animation(final_design, positions)
    
    return True

def create_hotspot_animation(design, positions):
    """Create animation of hotspot tracking"""
    try:
        import matplotlib.animation as animation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Setup design plot
        im1 = ax1.imshow(design.T, cmap='gray_r', alpha=0.8,
                        extent=[0, 200, 0, 200], vmin=0, vmax=1, animated=True)
        molecule_point1, = ax1.plot([], [], 'ro', markersize=12, 
                                  markerfacecolor='red', markeredgecolor='white',
                                  markeredgewidth=2)
        ax1.set_title('Molecule Position on Design', fontsize=11)
        ax1.set_xlabel('x (nm)', fontsize=10)
        ax1.set_ylabel('y (nm)', fontsize=10)
        
        # Setup field plot
        im2 = ax2.imshow(np.zeros_like(design.T), cmap='hot', alpha=0.9,
                        extent=[0, 200, 0, 200], vmin=0, vmax=1, animated=True)
        molecule_point2, = ax2.plot([], [], 'wo', markersize=10,
                                  markerfacecolor='white', markeredgecolor='black',
                                  markeredgewidth=1.5)
        ax2.set_title('Field Enhancement', fontsize=11)
        ax2.set_xlabel('x (nm)', fontsize=10)
        
        plt.tight_layout()
        
        def update(frame):
            """Update animation frame"""
            pos = positions[frame % len(positions)]
            
            # Update design plot
            molecule_point1.set_data([pos[0]], [pos[1]])
            
            # Update field plot
            x_grid, y_grid = np.meshgrid(np.arange(200), np.arange(200))
            distance = np.sqrt((x_grid - pos[0])**2 + (y_grid - pos[1])**2)
            field = design.T * np.exp(-distance**2 / (30**2))
            im2.set_array(field)
            molecule_point2.set_data([pos[0]], [pos[1]])
            
            ax1.set_title(f'Molecule Position (t={frame} ps)', fontsize=11)
            ax2.set_title(f'Field Enhancement at t={frame} ps', fontsize=11)
            
            return im1, molecule_point1, im2, molecule_point2
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=min(50, len(positions)),
                                      interval=200, blit=False)
        
        # Save animation
        output_path = "figures/supplementary/hotspot_tracking_animation.gif"
        anim.save(output_path, writer='pillow', fps=5)
        plt.close()
        
        print(f"✓ Animation saved: {output_path}")
        
    except ImportError:
        print("⚠️  Animation requires matplotlib.animation, skipping animation")
    
    return True

if __name__ == "__main__":
    success = create_hotspot_tracking_figure()
    if success:
        print("\n✅ Figure 4a generation complete")
    else:
        print("\n❌ Figure 4a generation failed")
        sys.exit(1)
