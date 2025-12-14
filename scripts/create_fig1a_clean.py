#!/usr/bin/env python3
"""
Create Figure 1a: Clean computational graph
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_clean_computational_graph():
    """Create clean, professional computational graph"""
    print("Creating CLEAN Figure 1a...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define nodes in two columns
    left_column = [
        (2, 7, "Design\nθ", "skyblue"),
        (2, 5, "Molecular\nDynamics", "lightcoral"),
        (2, 3, "Update\nθ ← θ - η∇ℒ", "lightgreen"),
    ]
    
    right_column = [
        (8, 8, "Material\nε(x,y)", "lightgreen"),
        (8, 6, "FDTD Solver\nMaxwell", "orange"),
        (8, 4, "E-field\nE(x,y,t)", "gold"),
        (12, 7, "Trajectory\nrₘ(t)", "pink"),
        (12, 5, "Interpolation\nE(rₘ)", "lightblue"),
        (12, 3, "Signal\nS = ∫|E|²dt", "lightyellow"),
        (16, 5, "Loss\nℒ(θ)", "red"),
        (16, 3, "Gradient\n∇ℒ", "violet"),
    ]
    
    # Draw left column nodes
    for x, y, label, color in left_column:
        circle = patches.Circle((x, y), 0.5, 
                               facecolor=color, edgecolor='black', 
                               linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold', wrap=True)
    
    # Draw right column nodes
    for x, y, label, color in right_column:
        rect = patches.Rectangle((x-1, y-0.4), 2, 0.8,
                                facecolor=color, edgecolor='black',
                                linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', wrap=True)
    
    # ARROW CONNECTIONS WITH CLEAR LABELS
    
    # Forward pass (blue, solid)
    connections = [
        # Design flow
        ((2, 7), (8, 8), "θ → ε", "blue", "solid"),
        ((8, 8), (8, 6), "ε → FDTD", "blue", "solid"),
        ((8, 6), (8, 4), "FDTD → E", "blue", "solid"),
        ((8, 4), (12, 5), "E → Interp.", "blue", "solid"),
        
        # Molecular flow
        ((2, 5), (12, 7), "MD → Traj.", "green", "solid"),
        ((12, 7), (12, 5), "Traj. → Interp.", "green", "solid"),
        
        # Signal flow
        ((12, 5), (12, 3), "Interp. → Signal", "blue", "solid"),
        ((12, 3), (16, 5), "Signal → Loss", "blue", "solid"),
        
        # Backward pass (red, dashed)
        ((16, 5), (16, 3), "Loss → ∇ℒ", "red", "dashed"),
        ((16, 3), (2, 3), "∇ℒ → Update", "red", "dashed"),
        ((2, 3), (2, 7), "Update → Design", "red", "dashed"),
    ]
    
    # Draw arrows
    for (x1, y1), (x2, y2), label, color, style in connections:
        # Arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color,
                                   linewidth=2, linestyle=style,
                                   alpha=0.8))
        
        # Label position (midpoint)
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        
        # Adjust label position to avoid overlap
        if abs(y1 - y2) < 0.1:  # Horizontal arrow
            ax.text(mx, my + 0.3, label, ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        else:  # Vertical or diagonal arrow
            ax.text(mx + 0.3, my, label, ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # Add phase labels
    ax.text(5, 7.5, "FORWARD PASS", fontsize=10, fontweight='bold',
           color='blue', ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue'))
    
    ax.text(9, 2.5, "BACKWARD PASS", fontsize=10, fontweight='bold',
           color='red', ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))
    
    # Add title
    ax.set_title("Figure 1a: Differentiable Computational Pipeline\n(Blue: Forward Simulation, Red: Gradient Backpropagation)", 
                fontsize=12, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(0, 18)
    ax.set_ylim(1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='skyblue', edgecolor='black', label='Design Parameters'),
        patches.Patch(facecolor='lightcoral', edgecolor='black', label='Molecular Simulation'),
        patches.Patch(facecolor='orange', edgecolor='black', label='Physics Engine'),
        patches.Patch(facecolor='lightblue', edgecolor='black', label='Data Processing'),
        patches.Patch(facecolor='red', edgecolor='black', label='Optimization'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9)
    
    # Save figure
    output_path = "figures/fig1a_computational_graph_clean.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CLEAN Figure 1a saved to: {output_path}")
    
    # Also save as the main figure name
    import shutil
    shutil.copy(output_path, "figures/fig1a_computational_graph.png")
    print("✓ Copied to: figures/fig1a_computational_graph.png")
    
    return True

if __name__ == "__main__":
    create_clean_computational_graph()
