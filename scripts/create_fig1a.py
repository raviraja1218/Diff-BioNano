#!/usr/bin/env python3
"""
Create Figure 1a: Computational graph of differentiable pipeline (Simplified)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_computational_graph():
    """Create computational graph figure without networkx"""
    print("Creating Figure 1a: Computational graph (simplified)...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Node positions (x, y, width, height, label, color)
    nodes = [
        (1, 7, 2, 1, "Design\nParameters\nθ", "lightblue"),
        (4, 7, 2, 1, "Material\nDistribution\nε(x,y)", "lightgreen"),
        (7, 8, 2, 1, "FDTD\nSolver", "orange"),
        (10, 8, 2, 1, "Electric\nField\nE(x,y,t)", "yellow"),
        (7, 6, 2, 1, "Molecular\nDynamics", "pink"),
        (10, 6, 2, 1, "Molecular\nTrajectory\nrₘ(t)", "lightpink"),
        (13, 7, 2, 1, "Field\nInterpolation\nE(rₘ(t))", "cyan"),
        (16, 7, 2, 1, "Signal\nCalculation\nS = ∫|E|²dt", "lightyellow"),
        (19, 7, 2, 1, "Loss\nFunction\nℒ(θ)", "red"),
        (16, 4, 2, 1, "Gradient\nCalculation\n∇ℒ", "purple"),
        (13, 4, 2, 1, "Parameter\nUpdate\nθ ← θ - η∇ℒ", "green"),
    ]
    
    # Draw nodes
    for x, y, w, h, label, color in nodes:
        rect = patches.Rectangle((x-w/2, y-h/2), w, h,
                                linewidth=2, edgecolor='black',
                                facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # Draw forward edges (blue)
    forward_connections = [
        (1, 7, 4, 7),   # Design → Material
        (4, 7, 7, 8),   # Material → FDTD
        (7, 8, 10, 8),  # FDTD → E-field
        (7, 6, 10, 6),  # MD → Trajectory
        (10, 8, 13, 7), # E-field → Interpolation
        (10, 6, 13, 7), # Trajectory → Interpolation
        (13, 7, 16, 7), # Interpolation → Signal
        (16, 7, 19, 7), # Signal → Loss
    ]
    
    for x1, y1, x2, y2 in forward_connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='blue',
                                   linewidth=2, alpha=0.8))
    
    # Draw backward edges (red, dashed)
    backward_connections = [
        (19, 7, 16, 4), # Loss → Gradient
        (16, 4, 13, 4), # Gradient → Update
        (13, 4, 1, 7),  # Update → Design (closing loop)
    ]
    
    for x1, y1, x2, y2 in backward_connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='red',
                                   linewidth=2, linestyle='dashed', alpha=0.8))
    
    # Add labels to edges
    ax.text(2.5, 7.2, "Forward", fontsize=8, color='blue', fontweight='bold')
    ax.text(17.5, 5.5, "Backward", fontsize=8, color='red', fontweight='bold')
    ax.text(7, 5.2, "Loop closure", fontsize=8, color='darkgreen', fontweight='bold')
    
    # Add title and legend
    ax.set_title("Figure 1a: Differentiable Computational Pipeline", 
                fontsize=14, fontweight='bold', pad=20)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Design Parameters'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Material Mapping'),
        Patch(facecolor='orange', edgecolor='black', label='Physics Engine'),
        Patch(facecolor='pink', edgecolor='black', label='Molecular Simulation'),
        Patch(facecolor='cyan', edgecolor='black', label='Data Processing'),
        Patch(facecolor='red', edgecolor='black', label='Optimization'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
              bbox_to_anchor=(0.02, 0.98))
    
    # Set limits and aspect
    ax.set_xlim(0, 21)
    ax.set_ylim(3, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add explanatory text
    ax.text(10.5, 3.3, "Blue arrows: Forward pass (simulation)", 
            fontsize=9, ha='center', color='blue')
    ax.text(10.5, 3.0, "Red dashed arrows: Backward pass (gradients)", 
            fontsize=9, ha='center', color='red')
    
    # Save figure
    output_path = "figures/fig1a_computational_graph.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 1a saved to: {output_path}")
    return True

if __name__ == "__main__":
    create_computational_graph()
