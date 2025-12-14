#!/usr/bin/env python3
"""
MINIMALIST Figure 1a: Computational Graph
Extremely simple, crystal clear
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_minimalist_fig1a():
    """Create minimalist computational graph"""
    print("Creating MINIMALIST Figure 1a: Computational Graph...")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Colors
    colors = {'blue': '#1f77b4', 'green': '#2ca02c', 'red': '#d62728', 
              'purple': '#9467bd', 'orange': '#ff7f0e'}
    
    # VERY SIMPLE NODES
    nodes_y = [6, 4, 2]  # Three rows
    
    # Row 1: Forward computation
    nodes1 = ['Design ε(x,y)', 'FDTD Solver', 'E-Field E(x,y,t)']
    for i, text in enumerate(nodes1):
        x = 1 + i * 3
        rect = patches.Rectangle((x-1.2, nodes_y[0]-0.3), 2.4, 0.6,
                                facecolor=colors['blue'], alpha=0.8,
                                edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, nodes_y[0], text, ha='center', va='center',
               color='white', fontsize=9, fontweight='bold')
        
        # Arrows between
        if i < len(nodes1)-1:
            ax.arrow(x+1.2, nodes_y[0], 0.6, 0, 
                    head_width=0.1, head_length=0.15,
                    fc=colors['blue'], ec=colors['blue'],
                    length_includes_head=True)
    
    # Row 2: Molecular data
    nodes2 = ['Molecular Dynamics', 'Trajectory rₘ(t)']
    for i, text in enumerate(nodes2):
        x = 1 + i * 4.5
        rect = patches.Rectangle((x-1.5, nodes_y[1]-0.3), 3.0, 0.6,
                                facecolor=colors['red'], alpha=0.8,
                                edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, nodes_y[1], text, ha='center', va='center',
               color='white', fontsize=9, fontweight='bold')
        
        if i == 0:
            ax.arrow(x+1.5, nodes_y[1], 1.5, 0, 
                    head_width=0.1, head_length=0.15,
                    fc=colors['red'], ec=colors['red'])
    
    # Row 3: Optimization
    nodes3 = ['Update (Adam)', 'Gradient ∂ℒ/∂ε', 'Loss ℒ']
    for i, text in enumerate(nodes3):
        x = 1 + i * 3
        rect = patches.Rectangle((x-1.2, nodes_y[2]-0.3), 2.4, 0.6,
                                facecolor=colors['purple'], alpha=0.8,
                                edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, nodes_y[2], text, ha='center', va='center',
               color='white', fontsize=9, fontweight='bold')
        
        if i < len(nodes3)-1:
            ax.arrow(x+1.2, nodes_y[2], 0.6, 0, 
                    head_width=0.1, head_length=0.15,
                    fc=colors['purple'], ec=colors['purple'],
                    linestyle='--')
    
    # Center: Signal processing
    signal_rect = patches.Rectangle((5-1.5, 3-0.3), 3.0, 0.6,
                                   facecolor=colors['green'], alpha=0.8,
                                   edgecolor='black')
    ax.add_patch(signal_rect)
    ax.text(5, 3, 'Signal ∫|E(rₘ)|²dt', ha='center', va='center',
           color='white', fontsize=9, fontweight='bold')
    
    # Connect rows with SIMPLE arrows
    # From E-Field to Signal (down)
    ax.arrow(7, 5.7, 0, -1.4, head_width=0.1, head_length=0.15,
            fc=colors['blue'], ec=colors['blue'])
    ax.text(7.2, 4.5, 'E(x,y,t)', fontsize=8, color=colors['blue'])
    
    # From Trajectory to Signal (right)
    ax.arrow(5.5, 4, 0, -0.7, head_width=0.1, head_length=0.15,
            fc=colors['red'], ec=colors['red'])
    ax.text(4.5, 3.5, 'rₘ(t)', fontsize=8, color=colors['red'])
    
    # From Signal to Loss (right-down)
    ax.arrow(6.5, 3, 0.8, -0.8, head_width=0.1, head_length=0.15,
            fc=colors['green'], ec=colors['green'])
    
    # Loop back (curved arrow)
    ax.annotate('', xy=(2, 2.3), xytext=(2, 5.7),
               arrowprops=dict(arrowstyle='->', color=colors['purple'],
                              linestyle='--', linewidth=1.5,
                              connectionstyle="arc3,rad=0.3"))
    ax.text(1.5, 4, 'ε_new', fontsize=8, color=colors['purple'])
    
    # Labels
    ax.text(5, 6.5, 'Electromagnetic Simulation', ha='center', va='center',
           fontsize=10, fontweight='bold', color=colors['blue'])
    ax.text(3, 4.5, 'Molecular Dynamics', ha='center', va='center',
           fontsize=10, fontweight='bold', color=colors['red'])
    ax.text(7, 1.5, 'Optimization', ha='center', va='center',
           fontsize=10, fontweight='bold', color=colors['purple'])
    
    ax.text(5, 0.8, 'Figure 1a: Differentiable Computational Framework',
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = "figures/fig1a_computational_graph_MINIMALIST.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ MINIMALIST Figure 1a saved to: {output_path}")
    return True

if __name__ == "__main__":
    create_minimalist_fig1a()
