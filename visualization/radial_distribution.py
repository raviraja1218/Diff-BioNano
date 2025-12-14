#!/usr/bin/env python3
"""
Generate Figure S6: Radial distribution analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_figure_S6():
    """Create Figure S6: Radial distribution analysis"""
    print("Creating Figure S6: Radial distribution analysis...")
    
    try:
        # Load trajectory
        trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
        positions = trajectory_data['positions']
        
        # Load final design
        final_design = np.load("data/optimization/final_design.npy")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    
    # Find hotspots in design (regions with design > 0.7)
    from scipy import ndimage
    
    # Label connected components of gold regions
    gold_mask = final_design > 0.7
    labeled_array, num_features = ndimage.label(gold_mask)
    
    # Calculate centers of mass for gold regions
    hotspots = []
    for i in range(1, num_features + 1):
        y_coords, x_coords = np.where(labeled_array == i)
        if len(x_coords) > 10:  # Only consider significant regions
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            hotspots.append((center_x, center_y))
    
    print(f"Found {len(hotspots)} hotspots in design")
    
    if len(hotspots) == 0:
        # Use design center as fallback
        hotspots = [(100, 100)]  # Center of 200×200 grid
    
    # Calculate distances from molecule to nearest hotspot
    distances = []
    for pos in positions[:1000]:  # Use first 1000 frames for speed
        x, y = pos
        min_dist = float('inf')
        for hx, hy in hotspots:
            dist = np.sqrt((x - hx)**2 + (y - hy)**2)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    
    distances = np.array(distances)
    
    # Generate random positions for comparison
    np.random.seed(42)
    random_positions = np.random.uniform(0, 200, (1000, 2))
    random_distances = []
    for pos in random_positions:
        x, y = pos
        min_dist = float('inf')
        for hx, hy in hotspots:
            dist = np.sqrt((x - hx)**2 + (y - hy)**2)
            if dist < min_dist:
                min_dist = dist
        random_distances.append(min_dist)
    
    random_distances = np.array(random_distances)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram of distances
    bins = np.linspace(0, 150, 31)
    
    ax1.hist(distances, bins=bins, alpha=0.7, color='crimson', 
             label='Molecule Positions', density=True)
    ax1.hist(random_distances, bins=bins, alpha=0.7, color='steelblue',
             label='Random Positions', density=True, histtype='step', linewidth=2)
    
    ax1.set_xlabel('Distance to Nearest Hotspot (pixels)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Distance Distribution', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate statistics
    mean_molecule = np.mean(distances)
    mean_random = np.mean(random_distances)
    std_molecule = np.std(distances)
    std_random = np.std(random_distances)
    
    ax1.axvline(mean_molecule, color='crimson', linestyle='--', linewidth=2)
    ax1.axvline(mean_random, color='steelblue', linestyle='--', linewidth=2)
    
    ax1.text(mean_molecule + 5, 0.025, f'Mean: {mean_molecule:.1f} px',
            fontsize=9, color='crimson', rotation=90)
    ax1.text(mean_random + 5, 0.025, f'Mean: {mean_random:.1f} px',
            fontsize=9, color='steelblue', rotation=90)
    
    # Right: Cumulative distribution
    from scipy.stats import gaussian_kde
    
    # KDE for smooth CDF
    kde_molecule = gaussian_kde(distances)
    kde_random = gaussian_kde(random_distances)
    
    x_eval = np.linspace(0, 150, 301)
    pdf_molecule = kde_molecule.evaluate(x_eval)
    pdf_random = kde_random.evaluate(x_eval)
    
    # Calculate CDF
    cdf_molecule = np.cumsum(pdf_molecule) / np.sum(pdf_molecule)
    cdf_random = np.cumsum(pdf_random) / np.sum(pdf_random)
    
    ax2.plot(x_eval, cdf_molecule, 'crimson', linewidth=3, label='Molecule')
    ax2.plot(x_eval, cdf_random, 'steelblue', linewidth=3, label='Random')
    
    ax2.set_xlabel('Distance to Nearest Hotspot (pixels)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution Function', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Mark 50% probability distance
    idx_50_molecule = np.argmax(cdf_molecule >= 0.5)
    idx_50_random = np.argmax(cdf_random >= 0.5)
    
    d50_molecule = x_eval[idx_50_molecule]
    d50_random = x_eval[idx_50_random]
    
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(d50_molecule, color='crimson', linestyle=':', alpha=0.7)
    ax2.axvline(d50_random, color='steelblue', linestyle=':', alpha=0.7)
    
    ax2.text(d50_molecule + 2, 0.55, f'D50: {d50_molecule:.1f} px',
            fontsize=9, color='crimson')
    ax2.text(d50_random + 2, 0.45, f'D50: {d50_random:.1f} px',
            fontsize=9, color='steelblue')
    
    # Calculate enhancement factor at short distances
    threshold = 50  # pixels
    prob_molecule_near = np.sum(distances < threshold) / len(distances)
    prob_random_near = np.sum(random_distances < threshold) / len(random_distances)
    
    if prob_random_near > 0:
        enhancement_factor = prob_molecule_near / prob_random_near
        ax2.text(100, 0.2, f'Enhancement (<{threshold} px): {enhancement_factor:.1f}×',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Supplementary Figure S6: Radial Distribution Analysis', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/supplementary/figS6_radial_distribution.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure S6 saved to: {output_path}")
    
    # Save radial analysis data
    radial_data = {
        'distances_molecule': distances,
        'distances_random': random_distances,
        'mean_molecule': mean_molecule,
        'mean_random': mean_random,
        'std_molecule': std_molecule,
        'std_random': std_random,
        'd50_molecule': d50_molecule,
        'd50_random': d50_random,
        'enhancement_factor': float(enhancement_factor) if 'enhancement_factor' in locals() else 1.0
    }
    
    np.save("data/optimization/analysis/radial_distribution.npy", radial_data)
    
    return True

if __name__ == "__main__":
    create_figure_S6()
