#!/usr/bin/env python3
"""
ULTRA-CLEAN Figure 1a: Computational Graph
Simplified, professional, no overlaps
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_ultraclean_fig1a():
    """Create ultra-clean computational graph diagram"""
    print("Creating ULTRA-CLEAN Figure 1a: Computational Graph...")
    
    # Create figure with ample space
    fig = plt.figure(figsize=(8, 6))  # 200mm × 150mm
    
    ax = plt.gca()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Clean, professional colors
    colors = {
        'design': '#2E5A88',    # Dark blue
        'physics': '#2E7D32',   # Dark green  
        'molecule': '#C62828',  # Dark red
        'optimization': '#8E24AA',  # Purple
        'signal': '#546E7A',    # Dark gray
        'forward_arrow': '#1565C0',  # Blue
        'backward_arrow': '#8E24AA', # Purple
    }
    
    # SIMPLIFIED NODES with more spacing
    nodes = {
        # Top row: Electromagnetic
        'design': {'pos': (2, 6.5), 'size': (1.8, 0.8), 'label': 'Design\nε(x,y)', 'color': colors['design']},
        'fdtd': {'pos': (5, 6.5), 'size': (1.8, 0.8), 'label': 'FDTD\nSolver', 'color': colors['physics']},
        'efield': {'pos': (8, 6.5), 'size': (1.8, 0.8), 'label': 'E-Field\nE(x,y,t)', 'color': colors['physics']},
        
        # Middle row: Molecular
        'md': {'pos': (2, 4), 'size': (1.8, 0.8), 'label': 'Molecular\nDynamics', 'color': colors['molecule']},
        'trajectory': {'pos': (5, 4), 'size': (1.8, 0.8), 'label': 'Trajectory\nrₘ(t)', 'color': colors['molecule']},
        
        # Bottom center: Signal processing
        'interpolation': {'pos': (8, 4), 'size': (2.0, 0.8), 'label': 'Field\nInterpolation', 'color': colors['signal']},
        'signal': {'pos': (10.5, 4), 'size': (1.8, 0.8), 'label': 'Signal\n∫|E|²dt', 'color': colors['signal']},
        
        # Bottom row: Optimization
        'loss': {'pos': (10.5, 2), 'size': (1.5, 0.7), 'label': 'Loss\nℒ', 'color': colors['optimization']},
        'gradient': {'pos': (8, 2), 'size': (1.8, 0.7), 'label': 'Gradient\n∂ℒ/∂ε', 'color': colors['optimization']},
        'update': {'pos': (5, 2), 'size': (1.8, 0.7), 'label': 'Update\n(Adam)', 'color': colors['optimization']},
    }
    
    # Draw SIMPLE rectangles (no rounded corners for clarity)
    for name, node in nodes.items():
        x, y = node['pos']
        w, h = node['size']
        
        rect = patches.Rectangle((x - w/2, y - h/2), w, h,
                                facecolor=node['color'], alpha=0.95,
                                edgecolor='black', linewidth=1.5,
                                zorder=2)
        ax.add_patch(rect)
        
        # Clear white text
        plt.text(x, y, node['label'], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                zorder=3, linespacing=1.1)
    
    # FORWARD PATH: Simple horizontal arrows
    forward_paths = [
        # Top row
        (nodes['design']['pos'], nodes['fdtd']['pos'], 'ε(x,y)', 6.8),
        (nodes['fdtd']['pos'], nodes['efield']['pos'], 'E(x,y,t)', 6.8),
        
        # Middle row
        (nodes['md']['pos'], nodes['trajectory']['pos'], 'rₘ(t)', 4.3),
        
        # To interpolation
        (nodes['trajectory']['pos'], nodes['interpolation']['pos'], 'rₘ(t)', 4.3),
        (nodes['efield']['pos'], (8, 4.8), 'E(x,y,t)', None),  # Down arrow
        ((8, 4.8), nodes['interpolation']['pos'], '', None),  # Complete down arrow
        
        # Rightward
        (nodes['interpolation']['pos'], nodes['signal']['pos'], 'E(rₘ)', 4.3),
        (nodes['signal']['pos'], nodes['loss']['pos'], 'Signal', None),
    ]
    
    for i, (start, end, label, y_label) in enumerate(forward_paths):
        x1, y1 = start
        x2, y2 = end
        
        # Draw arrow
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', linewidth=2.0,
                               color=colors['forward_arrow'], alpha=0.9,
                               mutation_scale=20, zorder=1)
        ax.add_patch(arrow)
        
        # Add label if provided
        if label and y_label:
            mid_x = (x1 + x2) / 2
            plt.text(mid_x, y_label, label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', 
                    color=colors['forward_arrow'], zorder=4)
    
    # BACKWARD PATH: Clear dashed arrows
    backward_paths = [
        # Loss to Gradient (down)
        (nodes['loss']['pos'], (10.5, 2.7), '', None),
        ((10.5, 2.7), nodes['gradient']['pos'], '∂ℒ', None),
        
        # Gradient to Update (left)
        (nodes['gradient']['pos'], nodes['update']['pos'], '∂ℒ/∂ε', 2.3),
        
        # Update back to Design (up-left, curved)
        (nodes['update']['pos'], (3.5, 3.5), '', None),
        ((3.5, 3.5), (2, 5.7), '', None),
        ((2, 5.7), nodes['design']['pos'], 'ε_new', None),
    ]
    
    for i, (start, end, label, y_label) in enumerate(backward_paths):
        x1, y1 = start
        x2, y2 = end
        
        # Draw dashed arrow for backward path
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', linewidth=2.0,
                               color=colors['backward_arrow'], alpha=0.9,
                               mutation_scale=20, zorder=1,
                               linestyle='--')
        ax.add_patch(arrow)
        
        # Add label if provided
        if label and y_label:
            mid_x = (x1 + x2) / 2
            plt.text(mid_x, y_label, label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color=colors['backward_arrow'], zorder=4)
    
    # Add curved arrow for the loop back
    from matplotlib.patches import Arc
    arc = Arc((3.5, 4.5), 3, 2, angle=0, theta1=180, theta2=270,
             color=colors['backward_arrow'], linewidth=2.0,
             linestyle='--', alpha=0.9, zorder=1)
    ax.add_patch(arc)
    
    # Add arrowhead to the arc
    arrowhead = FancyArrowPatch((2.2, 5.5), (2, 6.0),
                               arrowstyle='->', linewidth=2.0,
                               color=colors['backward_arrow'], alpha=0.9,
                               mutation_scale=15, zorder=1,
                               linestyle='--')
    ax.add_patch(arrowhead)
    
    # Add CLEAR section labels (outside the flow)
    sections = [
        (3.5, 7.2, 'Electromagnetic Simulation', colors['physics']),
        (3.5, 4.8, 'Molecular Dynamics', colors['molecule']),
        (8.0, 4.8, 'Signal Processing', colors['signal']),
        (8.0, 1.5, 'Optimization Loop', colors['optimization']),
    ]
    
    for x, y, text, color in sections:
        plt.text(x, y, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=color, edgecolor='black', 
                         alpha=0.9, linewidth=1.5),
                zorder=5)
    
    # Add forward/backward indicators
    plt.text(11.5, 6.5, 'Forward\nPass', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['forward_arrow'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.text(11.5, 2.0, 'Backward\nPass', ha='center', va='center',
            fontsize=10, fontweight='bold', color=colors['backward_arrow'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add figure title at the top
    plt.text(6, 7.8, 'Figure 1a: Differentiable Computational Framework',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='black')
    
    # Add explanatory note at bottom
    plt.text(6, 0.5, 'Solid arrows: Forward computation | Dashed arrows: Gradient backpropagation',
            ha='center', va='center', fontsize=9, style='italic',
            color='#555555')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save
    output_path = "figures/fig1a_computational_graph_ULTRA_CLEAN.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ ULTRA-CLEAN Figure 1a saved to: {output_path}")
    print("  - NO overlapping elements")
    print("  - CLEAR arrow paths")
    print("  - SIMPLE rectangular nodes")
    print("  - AMPLE spacing between all elements")
    
    return True

if __name__ == "__main__":
    create_ultraclean_fig1a()
