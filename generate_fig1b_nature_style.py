#!/usr/bin/env python3
"""
Generate Nature-style Figure 1b: Alanine Dipeptide Structure
Clean 3D visualization for publication
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_nature_style_molecule_figure():
    """Create publication-quality molecule figure"""
    print("Creating Nature-style Figure 1b: Alanine Dipeptide Structure...")
    
    # Create figure with proper dimensions
    fig = plt.figure(figsize=(6, 5))  # 150mm × 125mm
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Alanine dipeptide coordinates (ACE-ALA-NME)
    # Simplified backbone: N-Cα-C(O)-N-Cα-C(O)
    atoms = [
        # Backbone
        {'name': 'N', 'pos': [0.0, 0.0, 0.0], 'color': 'blue', 'size': 120},
        {'name': 'Cα', 'pos': [1.46, 0.0, 0.0], 'color': 'gray', 'size': 100},
        {'name': 'C', 'pos': [2.44, 1.43, 0.0], 'color': 'gray', 'size': 100},
        {'name': 'O', 'pos': [3.54, 1.43, 0.0], 'color': 'red', 'size': 110},
        # Side chain
        {'name': 'Cβ', 'pos': [1.46, -0.53, 1.23], 'color': 'gray', 'size': 90},
    ]
    
    # Bonds
    bonds = [
        (0, 1),  # N-Cα
        (1, 2),  # Cα-C
        (2, 3),  # C=O
        (1, 4),  # Cα-Cβ
    ]
    
    # Plot atoms
    for atom in atoms:
        ax.scatter(atom['pos'][0], atom['pos'][1], atom['pos'][2],
                  s=atom['size'], c=atom['color'], alpha=0.9,
                  edgecolors='black', linewidth=1.5, depthshade=True)
    
    # Plot bonds with cylinders (better than lines)
    for i, j in bonds:
        pos1 = atoms[i]['pos']
        pos2 = atoms[j]['pos']
        
        # Draw bond as cylinder
        x = [pos1[0], pos2[0]]
        y = [pos1[1], pos2[1]]
        z = [pos1[2], pos2[2]]
        
        ax.plot(x, y, z, color='black', linewidth=3, alpha=0.8)
    
    # Add atom labels with offsets
    label_offsets = {
        'N': [0.2, 0.2, 0.2],
        'Cα': [0.2, -0.2, -0.2],
        'C': [0.2, 0.2, 0.2],
        'O': [0.2, 0.2, -0.2],
        'Cβ': [0.2, -0.2, 0.2],
    }
    
    for atom in atoms:
        offset = label_offsets[atom['name']]
        ax.text(atom['pos'][0] + offset[0],
                atom['pos'][1] + offset[1],
                atom['pos'][2] + offset[2],
                atom['name'], fontsize=10, fontweight='bold')
    
    # Set labels and limits
    ax.set_xlabel('x (Å)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('y (Å)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('z (Å)', fontsize=11, fontweight='bold', labelpad=10)
    
    # Set limits for consistent view
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-1.0, 2.5)
    ax.set_zlim(-0.5, 2.0)
    
    # Set view angle for clarity
    ax.view_init(elev=25, azim=-60)
    
    # Add title
    ax.set_title('Alanine Dipeptide (ACE-ALA-NME)', fontsize=12, fontweight='bold', pad=20)
    
    # Add legend for atom types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Nitrogen (N)'),
        Patch(facecolor='gray', edgecolor='black', label='Carbon (C)'),
        Patch(facecolor='red', edgecolor='black', label='Oxygen (O)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add box with simulation info
    info_text = "Force Field: AMBER99SB-ILDN\nSolvent: GBN2 implicit\nTemperature: 300 K"
    ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes,
              fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save at Nature resolution
    output_path = "figures/fig1b_molecule_3d.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Nature-style Figure 1b saved to: {output_path}")
    print("  - 3D molecular structure with atom labels")
    print("  - Professional color coding (CPK scheme)")
    print("  - Clear bond representation")
    print("  - Simulation parameters included")
    
    return True

if __name__ == "__main__":
    create_nature_style_molecule_figure()

