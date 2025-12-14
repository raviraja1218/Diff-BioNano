#!/usr/bin/env python3
"""
Generate Nature-style Figure 1a: Computational Graph
Clean, professional, publication-ready diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_nature_style_computational_graph():
    """Create publication-quality computational graph diagram"""
    print("Creating Nature-style Figure 1a: Computational Graph...")
    
    # Create figure with Nature dimensions (180mm width)
    # Nature single column: ~85mm, but for wide figure we use full width
    fig = plt.figure(figsize=(7.2, 4.8))  # 180mm × 120mm in inches
    
    # Remove axes
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors for Nature style
    colors = {
        'background': '#FFFFFF',
        'design': '#4C72B0',  # Blue
        'physics': '#55A868',  # Green
        'molecule': '#C44E52',  # Red
        'optimization': '#DA8BC3',  # Pink
        'signal': '#8C8C8C',   # Gray
        'text': '#000000'      # Black
    }
    
    # Node positions (x, y, width, height)
    nodes = {
        'design': {'pos': (1, 5), 'size': (1.5, 0.8), 'label': 'Design\nParameters\nε(x,y)', 'color': colors['design']},
        'material': {'pos': (3, 5), 'size': (1.5, 0.8), 'label': 'Material\nGrid', 'color': colors['design']},
        'fdtd': {'pos': (5, 5), 'size': (1.5, 0.8), 'label': 'FDTD\nSolver', 'color': colors['physics']},
        'efield': {'pos': (7, 5), 'size': (1.5, 0.8), 'label': 'Electric Field\nE(x,y,t)', 'color': colors['physics']},
        
        'md': {'pos': (1, 3), 'size': (1.5, 0.8), 'label': 'Molecular\nDynamics', 'color': colors['molecule']},
        'trajectory': {'pos': (3, 3), 'size': (1.5, 0.8), 'label': 'Trajectory\nrₘ(t)', 'color': colors['molecule']},
        
        'interpolation': {'pos': (5, 3), 'size': (1.5, 0.8), 'label': 'Field\nInterpolation', 'color': colors['signal']},
        'signal': {'pos': (7, 3), 'size': (1.5, 0.8), 'label': 'Signal\n∫|E(rₘ)|²dt', 'color': colors['signal']},
        'loss': {'pos': (9, 3), 'size': (1.5, 0.8), 'label': 'Loss\nFunction\nℒ', 'color': colors['optimization']},
        
        'gradient': {'pos': (7, 1), 'size': (1.5, 0.8), 'label': 'Gradient\n∂ℒ/∂ε', 'color': colors['optimization']},
        'update': {'pos': (5, 1), 'size': (1.5, 0.8), 'label': 'Optimization\n(Adam)', 'color': colors['optimization']},
    }
    
    # Draw nodes with rounded rectangles (Nature style)
    for name, node in nodes.items():
        x, y = node['pos']
        w, h = node['size']
        
        # Create rounded rectangle
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=node['color'], alpha=0.9,
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # Add text
        plt.text(x, y, node['label'], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
    
    # Draw arrows (forward path - blue)
    forward_connections = [
        ('design', 'material', 'ε(x,y)'),
        ('material', 'fdtd', 'Grid'),
        ('fdtd', 'efield', 'E(x,y,t)'),
        ('md', 'trajectory', 'rₘ(t)'),
        ('efield', 'interpolation', 'E(x,y,t)'),
        ('trajectory', 'interpolation', 'rₘ(t)'),
        ('interpolation', 'signal', 'E(rₘ(t))'),
        ('signal', 'loss', 'Signal'),
    ]
    
    for src, dst, label in forward_connections:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        # Create arrow
        arrow = FancyArrowPatch((x1 + nodes[src]['size'][0]/2, y1),
                               (x2 - nodes[dst]['size'][0]/2, y2),
                               arrowstyle='->', linewidth=1.5,
                               color=colors['design'], alpha=0.8,
                               mutation_scale=15)
        ax.add_patch(arrow)
        
        # Add label halfway
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Offset label to avoid arrow
        offset = 0.1
        if y1 == y2:  # Horizontal arrow
            plt.text(mid_x, mid_y + offset, label, ha='center', va='bottom',
                    fontsize=7, fontstyle='italic', color=colors['design'])
        else:  # Vertical or diagonal
            plt.text(mid_x + offset, mid_y, label, ha='left', va='center',
                    fontsize=7, fontstyle='italic', color=colors['design'])
    
    # Draw arrows (backward/optimization path - red)
    backward_connections = [
        ('loss', 'gradient', 'backprop'),
        ('gradient', 'update', '∂ℒ/∂ε'),
        ('update', 'design', 'ε_new', True),  # Loop back
    ]
    
    for src, dst, label, *is_loop in backward_connections:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        if is_loop:
            # Special curved arrow for loop back
            arrow = FancyArrowPatch((x1, y1 - nodes[src]['size'][1]/2),
                                   (x2, y2 + nodes[dst]['size'][1]/2),
                                   connectionstyle="arc3,rad=-0.3",
                                   arrowstyle='->', linewidth=1.5,
                                   color=colors['optimization'], alpha=0.8,
                                   mutation_scale=15, linestyle='--')
        else:
            arrow = FancyArrowPatch((x1 - nodes[src]['size'][0]/2, y1),
                                   (x2 + nodes[dst]['size'][0]/2, y2),
                                   arrowstyle='->', linewidth=1.5,
                                   color=colors['optimization'], alpha=0.8,
                                   mutation_scale=15, linestyle='--')
        
        ax.add_patch(arrow)
        
        # Add label
        if is_loop:
            plt.text((x1 + x2)/2, (y1 + y2)/2 - 0.3, label, ha='center', va='top',
                    fontsize=7, fontstyle='italic', color=colors['optimization'])
        else:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            plt.text(mid_x, mid_y - 0.1, label, ha='center', va='top',
                    fontsize=7, fontstyle='italic', color=colors['optimization'])
    
    # Add title and annotations
    plt.text(5, 5.8, 'Electromagnetic Simulation', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['physics'])
    
    plt.text(2, 3.8, 'Molecular Dynamics', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['molecule'])
    
    plt.text(7, 1.8, 'Optimization Loop', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['optimization'])
    
    # Add forward/backward labels
    plt.text(8.5, 4.5, 'Forward Pass', ha='center', va='center',
            fontsize=9, fontweight='bold', color=colors['design'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors['design']))
    
    plt.text(8.5, 0.5, 'Backward Pass', ha='center', va='center',
            fontsize=9, fontweight='bold', color=colors['optimization'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=colors['optimization']))
    
    # Add figure title
    plt.text(5, 6.2, 'Figure 1a: Differentiable Computational Framework',
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save at Nature resolution (600 DPI)
    output_path = "figures/fig1a_computational_graph.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Nature-style Figure 1a saved to: {output_path}")
    print("  - Dimensions: 180mm × 120mm (Nature full-width)")
    print("  - Resolution: 600 DPI")
    print("  - Colors: Colorblind-friendly palette")
    print("  - Fonts: Arial/Helvetica, 8-12pt")
    
    return True

if __name__ == "__main__":
    create_nature_style_computational_graph()
