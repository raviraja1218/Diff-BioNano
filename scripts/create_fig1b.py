#!/usr/bin/env python3
"""
Create Figure 1b: Alanine dipeptide molecular structure
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_molecule_structure():
    """Create 3D molecular structure figure"""
    print("Creating Figure 1b: Alanine dipeptide structure...")
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Alanine dipeptide coordinates (simplified ACE-ALA-NME)
    # Backbone atoms: N, CA, C, O, CB
    atoms = ['N', 'Cα', 'C', 'O', 'Cβ']
    x = [0.0, 1.5, 2.5, 3.5, 1.5]
    y = [0.0, 0.0, 1.4, 1.4, -0.5]
    z = [0.0, 0.0, 0.0, 0.0, 1.2]
    
    # Atom colors (CPK coloring)
    colors = ['blue', 'gray', 'gray', 'red', 'gray']
    sizes = [400, 300, 300, 350, 250]
    
    # Plot atoms
    for i in range(len(atoms)):
        ax.scatter(x[i], y[i], z[i], s=sizes[i], c=colors[i], 
                  alpha=0.9, edgecolors='black', linewidth=1.5)
    
    # Plot bonds
    bonds = [(0, 1), (1, 2), (2, 3), (1, 4)]  # N-CA, CA-C, C-O, CA-CB
    for i, j in bonds:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
               'k-', linewidth=3, alpha=0.8)
    
    # Labels
    ax.text(x[0]-0.3, y[0], z[0], 'N', fontsize=11, fontweight='bold', color='blue')
    ax.text(x[1]+0.2, y[1], z[1], 'Cα', fontsize=11, fontweight='bold', color='black')
    ax.text(x[2]+0.2, y[2], z[2], 'C', fontsize=11, fontweight='bold', color='black')
    ax.text(x[3]+0.2, y[3], z[3], 'O', fontsize=11, fontweight='bold', color='red')
    ax.text(x[4]-0.3, y[4], z[4], 'Cβ', fontsize=11, fontweight='bold', color='black')
    
    # Add backbone label
    ax.text(1.0, 0.0, -0.5, 'Backbone', fontsize=10, color='darkblue', fontweight='bold')
    ax.text(1.5, -0.8, 1.2, 'Sidechain', fontsize=10, color='darkgreen', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X (Å)', fontsize=11, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=11, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=11, labelpad=10)
    ax.set_title('Figure 1b: Alanine Dipeptide Structure (ACE-ALA-NME)', 
                fontsize=12, fontweight='bold', pad=20)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
    mid_x = (max(x) + min(x)) * 0.5
    mid_y = (max(y) + min(y)) * 0.5
    mid_z = (max(z) + min(z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save figure
    output_path = "figures/fig1b_molecule_3d.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 1b saved to: {output_path}")
    
    # Also create a simple 2D version
    fig2d, ax2d = plt.subplots(figsize=(6, 6))
    
    # 2D projection
    ax2d.scatter(x, y, s=[s*0.7 for s in sizes], c=colors, 
                alpha=0.9, edgecolors='black', linewidth=1.5)
    
    for i, j in bonds:
        ax2d.plot([x[i], x[j]], [y[i], y[j]], 'k-', linewidth=3, alpha=0.8)
    
    for i, atom in enumerate(atoms):
        ax2d.text(x[i]+0.1, y[i]+0.1, atom, fontsize=10, fontweight='bold')
    
    ax2d.set_xlabel('X (Å)', fontsize=11)
    ax2d.set_ylabel('Y (Å)', fontsize=11)
    ax2d.set_title('Alanine Dipeptide - Top View', fontsize=12, fontweight='bold')
    ax2d.grid(True, alpha=0.3)
    ax2d.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("figures/fig1b_molecule_2d.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print("✓ Created 2D version: figures/fig1b_molecule_2d.png")
    
    return True

if __name__ == "__main__":
    create_molecule_structure()
