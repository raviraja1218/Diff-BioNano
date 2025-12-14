#!/usr/bin/env python3
"""
FIXED: Nature-style Figure 1a: Computational Graph
Clear arrows and text, no overlaps
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_fixed_fig1a():
    """Create FIXED publication-quality computational graph"""
    print("Creating FIXED Nature-style Figure 1a: Computational Graph...")
    
    # Create figure with more vertical space
    fig = plt.figure(figsize=(7.5, 5.5))  # 188mm × 138mm
    
    ax = plt.gca()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Professional color palette
    colors = {
        'design': '#2E5A88',    # Darker blue
        'physics': '#2E7D32',   # Darker green
        'molecule': '#C62828',  # Darker red
        'optimization': '#8E24AA',  # Purple
        'signal': '#455A64',    # Dark gray
        'arrow_forward': '#1565C0',  # Bright blue
        'arrow_backward': '#8E24AA', # Purple
    }
    
    # Node positions with more spacing
    nodes = {
        'design': {'pos': (1.5, 6), 'size': (1.6, 0.9), 'label': 'Design\nParameters\nε(x,y)', 'color': colors['design']},
        'material': {'pos': (4.0, 6), 'size': (1.6, 0.9), 'label': 'Material\nGrid', 'color': colors['design']},
        'fdtd': {'pos': (6.5, 6), 'size': (1.6, 0.9), 'label': 'FDTD\nSolver', 'color': colors['physics']},
        'efield': {'pos': (9.0, 6), 'size': (1.6, 0.9), 'label': 'E-Field\nE(x,y,t)', 'color': colors['physics']},
        
        'md': {'pos': (1.5, 4), 'size': (1.6, 0.9), 'label': 'Molecular\nDynamics', 'color': colors['molecule']},
        'trajectory': {'pos': (4.0, 4), 'size': (1.6, 0.9), 'label': 'Trajectory\nrₘ(t)', 'color': colors['molecule']},
        
        'interpolation': {'pos': (6.5, 4), 'size': (1.8, 0.9), 'label': 'Field\nInterpolation', 'color': colors['signal']},
        'signal': {'pos': (9.0, 4), 'size': (1.6, 0.9), 'label': 'Signal\n∫|E(rₘ)|²dt', 'color': colors['signal']},
        'loss': {'pos': (10.5, 4), 'size': (1.6, 0.9), 'label': 'Loss\nℒ', 'color': colors['optimization']},
        
        'gradient': {'pos': (9.0, 2), 'size': (1.6, 0.9), 'label': 'Gradient\n∂ℒ/∂ε', 'color': colors['optimization']},
        'update': {'pos': (6.5, 2), 'size': (1.8, 0.9), 'label': 'Optimizer\n(Adam)', 'color': colors['optimization']},
    }
    
    # Draw nodes
    for name, node in nodes.items():
        x, y = node['pos']
        w, h = node['size']
        
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.12",
                             facecolor=node['color'], alpha=0.95,
                             edgecolor='black', linewidth=1.8)
        ax.add_patch(rect)
        
        plt.text(x, y, node['label'], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                linespacing=1.2)
    
    # FORWARD ARROWS (Horizontal, clear labels)
    forward_arrows = [
        ('design', 'material', (2.8, 6.15), 'ε(x,y)'),
        ('material', 'fdtd', (5.25, 6.15), 'Grid'),
        ('fdtd', 'efield', (7.75, 6.15), 'E(x,y,t)'),
        ('md', 'trajectory', (2.8, 4.15), 'rₘ(t)'),
    ]
    
    for src, dst, label_pos, label in forward_arrows:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        # Straight horizontal arrow
        arrow = FancyArrowPatch((x1 + nodes[src]['size'][0]/2, y1),
                               (x2 - nodes[dst]['size'][0]/2, y2),
                               arrowstyle='->', linewidth=2.2,
                               color=colors['arrow_forward'], alpha=0.9,
                               mutation_scale=18)
        ax.add_patch(arrow)
        
        # Label above arrow
        plt.text(label_pos[0], label_pos[1], label, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=colors['arrow_forward'])
    
    # DOWNWARD ARROWS (Vertical, clear labels)
    downward_arrows = [
        ('efield', 'interpolation', (9.0, 5.4), 'E(x,y,t)', 'down'),
        ('trajectory', 'interpolation', (4.0, 4.9), 'rₘ(t)', 'down'),
    ]
    
    for src, dst, label_pos, label, direction in downward_arrows:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        # Vertical arrow
        if direction == 'down':
            arrow = FancyArrowPatch((x1, y1 - nodes[src]['size'][1]/2),
                                   (x2, y2 + nodes[dst]['size'][1]/2),
                                   arrowstyle='->', linewidth=2.2,
                                   color=colors['arrow_forward'], alpha=0.9,
                                   mutation_scale=18)
        
        ax.add_patch(arrow)
        
        # Label to the right of arrow
        plt.text(label_pos[0] + 0.3, (y1 + y2)/2, label, ha='left', va='center',
                fontsize=8, fontweight='bold', color=colors['arrow_forward'])
    
    # RIGHTWARD ARROWS in middle row
    middle_arrows = [
        ('interpolation', 'signal', (7.65, 4.15), 'E(rₘ(t))'),
        ('signal', 'loss', (9.75, 4.15), 'Signal'),
    ]
    
    for src, dst, label_pos, label in middle_arrows:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        arrow = FancyArrowPatch((x1 + nodes[src]['size'][0]/2, y1),
                               (x2 - nodes[dst]['size'][0]/2, y2),
                               arrowstyle='->', linewidth=2.2,
                               color=colors['arrow_forward'], alpha=0.9,
                               mutation_scale=18)
        ax.add_patch(arrow)
        
        plt.text(label_pos[0], label_pos[1], label, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=colors['arrow_forward'])
    
    # BACKWARD ARROWS (Optimization loop) - CLEAR
    backward_arrows = [
        ('loss', 'gradient', (10.5, 3.4), '∂ℒ', 'down'),
        ('gradient', 'update', (7.75, 2.15), '∂ℒ/∂ε', 'left'),
        ('update', 'design', (4.0, 2.8), 'ε_new', 'up-left'),
    ]
    
    for src, dst, label_pos, label, direction in backward_arrows:
        x1, y1 = nodes[src]['pos']
        x2, y2 = nodes[dst]['pos']
        
        if direction == 'down':
            arrow = FancyArrowPatch((x1, y1 - nodes[src]['size'][1]/2),
                                   (x2, y2 + nodes[dst]['size'][1]/2),
                                   arrowstyle='->', linewidth=2.2,
                                   color=colors['arrow_backward'], alpha=0.9,
                                   mutation_scale=18, linestyle='--')
        elif direction == 'left':
            arrow = FancyArrowPatch((x1 - nodes[src]['size'][0]/2, y1),
                                   (x2 + nodes[dst]['size'][0]/2, y2),
                                   arrowstyle='->', linewidth=2.2,
                                   color=colors['arrow_backward'], alpha=0.9,
                                   mutation_scale=18, linestyle='--')
        else:  # up-left (loop back)
            arrow = FancyArrowPatch((x1 - nodes[src]['size'][0]/2, y1),
                                   (x2, y2 + nodes[dst]['size'][1]/2),
                                   connectionstyle="arc3,rad=0.2",
                                   arrowstyle='->', linewidth=2.2,
                                   color=colors['arrow_backward'], alpha=0.9,
                                   mutation_scale=18, linestyle='--')
        
        ax.add_patch(arrow)
        
        # Label placement
        if direction == 'down':
            plt.text(label_pos[0] + 0.3, (y1 + y2)/2, label, ha='left', va='center',
                    fontsize=8, fontweight='bold', color=colors['arrow_backward'])
        elif direction == 'left':
            plt.text(label_pos[0], label_pos[1], label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=colors['arrow_backward'])
        else:
            plt.text(label_pos[0], label_pos[1], label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=colors['arrow_backward'])
    
    # Add section labels with boxes
    section_boxes = [
        (5.5, 6.5, 'Electromagnetic Simulation', colors['physics']),
        (2.75, 4.5, 'Molecular Dynamics', colors['molecule']),
        (8.5, 1.5, 'Optimization Loop', colors['optimization']),
    ]
    
    for x, y, text, color in section_boxes:
        plt.text(x, y, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=color, edgecolor='black', alpha=0.9))
    
    # Add forward/backward labels
    plt.text(10.8, 5.2, 'Forward Pass', ha='right', va='center',
            fontsize=10, fontweight='bold', color=colors['arrow_forward'],
            rotation=90)
    
    plt.text(10.8, 3.0, 'Backward Pass', ha='right', va='center',
            fontsize=10, fontweight='bold', color=colors['arrow_backward'],
            rotation=90)
    
    # Add figure title
    plt.text(5.5, 7.2, 'Figure 1a: Differentiable Computational Framework',
            ha='center', va='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save at Nature resolution
    output_path = "figures/fig1a_computational_graph_FIXED.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ FIXED Figure 1a saved to: {output_path}")
    print("  - Clear, non-overlapping arrows")
    print("  - Readable text labels")
    print("  - Proper spacing between elements")
    print("  - Professional color scheme")
    
    return True

if __name__ == "__main__":
    create_fixed_fig1a()
