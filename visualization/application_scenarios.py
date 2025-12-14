#!/usr/bin/env python3
"""
Generate Figure 5c: Application scenarios
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Arrow
from matplotlib.patches import Wedge, Polygon
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_figure_5c():
    """Create Figure 5c: Application scenarios"""
    print("Creating Figure 5c: Application scenarios...")
    
    fig = plt.figure(figsize=(14, 5))
    
    # Three application scenarios
    applications = [
        {
            "title": "Single-Molecule Detection",
            "description": "Ultra-sensitive detection of low-abundance biomarkers",
            "features": ["< 1 fM sensitivity", "Real-time monitoring", "Label-free"],
            "color": "royalblue"
        },
        {
            "title": "Real-Time Folding Studies",
            "description": "Monitor protein folding/unfolding dynamics",
            "features": ["µs time resolution", "Native conditions", "No immobilization"],
            "color": "forestgreen"
        },
        {
            "title": "Point-of-Care Diagnostics",
            "description": "Portable biosensing for clinical applications",
            "features": ["Smartphone readout", "< 10 min assay", "Multiplexed detection"],
            "color": "darkorange"
        }
    ]
    
    for idx, app in enumerate(applications):
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw application schematic
        center_x, center_y = 5, 5
        
        # Background sensor chip
        chip = FancyBboxPatch((1, 1), 8, 8,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgray', edgecolor='black',
                             alpha=0.3, linewidth=2)
        ax.add_patch(chip)
        
        # Our optimized sensor (stylized)
        sensor_center = (center_x, center_y + 1)
        
        # Draw chiral sensor pattern
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i, angle in enumerate(angles):
            radius = 2.0
            wedge = Wedge(sensor_center, radius, 
                         np.degrees(angle), np.degrees(angle + np.pi/3),
                         facecolor=app['color'], alpha=0.7, 
                         edgecolor='black', linewidth=1)
            ax.add_patch(wedge)
            
            # Add some asymmetry for chirality
            if i % 2 == 0:
                inner_wedge = Wedge(sensor_center, 1.2,
                                   np.degrees(angle + 0.1), np.degrees(angle + np.pi/3 - 0.1),
                                   facecolor='gold', alpha=0.9,
                                   edgecolor='black', linewidth=0.5)
                ax.add_patch(inner_wedge)
        
        # Molecule above sensor
        molecule_y = center_y + 3.5
        molecule = Circle((center_x, molecule_y), 0.6,
                         facecolor='purple', edgecolor='black',
                         linewidth=1.5, alpha=0.8)
        ax.add_patch(molecule)
        
        # Dynamic motion lines
        motion_amplitude = 0.8
        motion_points = np.array([
            [center_x, molecule_y],
            [center_x + motion_amplitude*np.sin(0), molecule_y + motion_amplitude*np.cos(0)],
            [center_x + motion_amplitude*np.sin(np.pi/2), molecule_y + motion_amplitude*np.cos(np.pi/2)],
            [center_x + motion_amplitude*np.sin(np.pi), molecule_y + motion_amplitude*np.cos(np.pi)],
            [center_x + motion_amplitude*np.sin(3*np.pi/2), molecule_y + motion_amplitude*np.cos(3*np.pi/2)],
            [center_x, molecule_y]
        ])
        motion_line = Polygon(motion_points, closed=True,
                             fill=False, edgecolor='red',
                             linewidth=1.5, alpha=0.6, linestyle='--')
        ax.add_patch(motion_line)
        
        # Signal detection arrows
        arrow_start = (center_x, center_y - 2)
        arrow_end = (center_x, 0.5)
        arrow = Arrow(arrow_start[0], arrow_start[1],
                     arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                     width=0.3, facecolor='red', edgecolor='darkred')
        ax.add_patch(arrow)
        
        # Detection device (simplified)
        if idx == 0:  # Spectrometer
            detector = Rectangle((center_x - 1, 0), 2, 0.4,
                                facecolor='silver', edgecolor='black',
                                linewidth=1)
            ax.add_patch(detector)
            ax.text(center_x, 0.2, 'Spectrometer', ha='center', va='center',
                   fontsize=8, fontweight='bold')
        elif idx == 1:  # Microscope
            detector = FancyBboxPatch((center_x - 1.2, 0), 2.4, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor='silver', edgecolor='black',
                                     linewidth=1)
            ax.add_patch(detector)
            ax.text(center_x, 0.25, 'Microscope', ha='center', va='center',
                   fontsize=8, fontweight='bold')
        else:  # Smartphone
            detector = Rectangle((center_x - 1.5, 0), 3, 0.6,
                                facecolor='black', edgecolor='gray',
                                linewidth=2)
            ax.add_patch(detector)
            ax.text(center_x, 0.3, '📱', ha='center', va='center', fontsize=12)
        
        # Application title
        ax.text(center_x, 9.2, app['title'], ha='center', va='center',
               fontsize=11, fontweight='bold', color=app['color'])
        
        # Description
        ax.text(center_x, 8.4, app['description'], ha='center', va='center',
               fontsize=9, style='italic', wrap=True)
        
        # Features
        features_y = 7.2
        for feature in app['features']:
            ax.text(center_x, features_y, f"• {feature}", 
                   ha='center', va='center', fontsize=8)
            features_y -= 0.6
    
    plt.suptitle('Figure 5c: Application Scenarios', fontsize=14, y=0.95)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/results/fig5c_applications.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 5c saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    create_figure_5c()
