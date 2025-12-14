#!/usr/bin/env python3
"""
FIXED: Nature-style Figure 1b: Alanine Dipeptide Structure
Clean 3D visualization without overlapping elements
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

def create_fixed_fig1b():
    """Create FIXED publication-quality molecule figure"""
    print("Creating FIXED Nature-style Figure 1b: Alanine Dipeptide Structure...")
    
    # Create figure with proper dimensions
    fig = plt.figure(figsize=(6.5, 5.5))  # 163mm × 138mm
    
    # 3D plot with more space
    ax = fig.add_subplot(111, projection='3d')
    
    # Alanine dipeptide coordinates (ACE-ALA-NME)
    atoms = [
        # Backbone
        {'name': 'N', 'pos': [0.0, 0.0, 0.0], 'color': 'blue', 'size': 180},
        {'name': 'Cα', 'pos': [1.46, 0.0, 0.0], 'color': '#404040', 'size': 160},
        {'name': 'C', 'pos': [2.44, 1.43, 0.0], 'color': '#404040', 'size': 160},
        {'name': 'O', 'pos': [3.54, 1.43, 0.0], 'color': 'red', 'size': 170},
        # Side chain
        {'name': 'Cβ', 'pos': [1.46, -0.53, 1.23], 'color': '#404040', 'size': 150},
        # Additional atoms for completeness
        {'name': 'Hα', 'pos': [1.46, 0.99, -0.54], 'color': 'white', 'size': 80},
        {'name': 'Hβ1', 'pos': [2.37, -0.53, 1.23], 'color': 'white', 'size': 80},
        {'name': 'Hβ2', 'pos': [0.96, -1.43, 1.23], 'color': 'white', 'size': 80},
    ]
    
    # Bonds
    bonds = [
        (0, 1),  # N-Cα
        (1, 2),  # Cα-C
        (2, 3),  # C=O
        (1, 4),  # Cα-Cβ
        (1, 5),  # Cα-Hα
        (4, 6),  # Cβ-Hβ1
        (4, 7),  # Cβ-Hβ2
    ]
    
    # Plot atoms
    for atom in atoms:
        ax.scatter(atom['pos'][0], atom['pos'][1], atom['pos'][2],
                  s=atom['size'], c=atom['color'], alpha=0.95,
                  edgecolors='black', linewidth=1.8, depthshade=True,
                  zorder=10)
    
    # Plot bonds with better visibility
    for i, j in bonds:
        pos1 = atoms[i]['pos']
        pos2 = atoms[j]['pos']
        
        x = [pos1[0], pos2[0]]
        y = [pos1[1], pos2[1]]
        z = [pos1[2], pos2[2]]
        
        # Thicker bonds for backbone, thinner for hydrogens
        linewidth = 4.0 if i < 5 and j < 5 else 2.5
        ax.plot(x, y, z, color='black', linewidth=linewidth, alpha=0.9, zorder=5)
    
    # Add atom labels with careful positioning
    label_positions = {
        'N': [0.0, 0.0, 0.35],   # Above nitrogen
        'Cα': [1.46, -0.3, -0.3],  # Below and left
        'C': [2.44, 1.73, 0.0],   # Above carbon
        'O': [3.54, 1.73, 0.0],   # Above oxygen
        'Cβ': [1.46, -0.53, 1.53], # Above Cβ
    }
    
    for atom in atoms[:5]:  # Only label heavy atoms
        if atom['name'] in label_positions:
            pos = label_positions[atom['name']]
            ax.text(pos[0], pos[1], pos[2], 
                   atom['name'], fontsize=12, fontweight='bold',
                   ha='center', va='center', zorder=15)
    
    # Set labels and limits - MORE SPACE
    ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold', labelpad=15)
    
    # Set limits with more margin
    ax.set_xlim(-0.8, 4.2)
    ax.set_ylim(-1.5, 2.8)
    ax.set_zlim(-0.8, 2.3)
    
    # Improve tick labels
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_zticks([0, 1, 2])
    
    # Set view angle for optimal clarity
    ax.view_init(elev=20, azim=-50)
    
    # Add title with more space
    plt.suptitle('Alanine Dipeptide Structure (ACE-ALA-NME)', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Add clean legend outside the plot
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Nitrogen (N)'),
        Patch(facecolor='#404040', edgecolor='black', label='Carbon (C)'),
        Patch(facecolor='red', edgecolor='black', label='Oxygen (O)'),
        Patch(facecolor='white', edgecolor='black', label='Hydrogen (H)'),
    ]
    
    # Place legend in empty space
    ax.legend(handles=legend_elements, loc='upper left', 
              fontsize=10, framealpha=0.95, edgecolor='black',
              bbox_to_anchor=(0.02, 0.98))  # Top-left corner
    
    # Add simulation info in top-right corner (NOT covering plot)
    info_text = "Simulation Parameters:\n• AMBER99SB-ILDN\n• GBN2 implicit solvent\n• 300 K, NVT ensemble"
    ax.text2D(0.98, 0.98, info_text, transform=ax.transAxes,
              fontsize=9, ha='right', va='top',
              bbox=dict(boxstyle='round', facecolor='white', 
                       alpha=0.9, edgecolor='gray', pad=5))
    
    # Adjust layout to prevent clipping
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.90)
    
    # Save at Nature resolution
    output_path = "figures/fig1b_molecule_3d_FIXED.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ FIXED Figure 1b saved to: {output_path}")
    print("  - No overlapping elements")
    print("  - Clear axis labels")
    print("  - Proper legend placement")
    print("  - Clean information box")
    
    return True

if __name__ == "__main__":
    create_fixed_fig1b()
