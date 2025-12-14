#!/usr/bin/env python3
"""
Experiment 1: Monte Carlo fabrication analysis
Simulate 1000 devices with random fabrication errors
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import ndimage
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def monte_carlo_fabrication_analysis():
    """Simulate fabrication of 1000 devices with random errors"""
    print("Running Experiment 1: Monte Carlo fabrication analysis...")
    
    # Load final design
    final_design = np.load("data/optimization/final_design.npy")
    grid_size = final_design.shape
    
    # Parameters
    n_devices = 1000
    performances = []
    
    print(f"Simulating {n_devices} fabricated devices...")
    
    for i in range(n_devices):
        if i % 100 == 0:
            print(f"  Device {i}/{n_devices}")
        
        # Random fabrication errors (typical e-beam values)
        feature_error_std = 10  # nm (typical e-beam placement error)
        linewidth_error_std = 5  # nm (typical linewidth variation)
        edge_roughness_std = 3  # nm (typical edge roughness)
        
        # Create error mask
        error_mask = np.random.normal(0, 1, grid_size)
        
        # Apply Gaussian blur for spatial correlation (errors are correlated)
        from scipy.ndimage import gaussian_filter
        error_mask = gaussian_filter(error_mask, sigma=2)
        
        # Scale to actual error magnitudes
        placement_error = error_mask * feature_error_std / 1  # 1 nm per pixel
        
        # Create "fabricated" design
        fabricated_design = np.zeros_like(final_design)
        
        # Threshold with placement error
        threshold = 0.5 + placement_error / 100  # Small threshold shift
        
        # Binary design with errors
        fabricated_design = np.where(final_design > threshold, 1.0, 0.0)
        
        # Add edge roughness
        if edge_roughness_std > 0:
            roughness = np.random.normal(0, edge_roughness_std/1, grid_size)
            roughness = gaussian_filter(roughness, sigma=1)
            fabricated_design = np.clip(fabricated_design + roughness/100, 0, 1)
        
        # Calculate performance relative to ideal
        # Simple performance metric: overlap with ideal design
        ideal_gold = final_design > 0.7
        fabricated_gold = fabricated_design > 0.5
        
        # Performance = overlap / ideal area
        if np.sum(ideal_gold) > 0:
            overlap = np.sum(ideal_gold & fabricated_gold)
            performance = overlap / np.sum(ideal_gold)
        else:
            performance = 0.0
        
        performances.append(performance)
    
    performances = np.array(performances)
    
    # Calculate statistics
    mean_performance = np.mean(performances)
    std_performance = np.std(performances)
    yield_80 = np.mean(performances > 0.8) * 100  # % with >80% performance
    yield_90 = np.mean(performances > 0.9) * 100  # % with >90% performance
    
    print(f"\n=== Monte Carlo Results (N={n_devices}) ===")
    print(f"Mean performance: {mean_performance:.3f} ± {std_performance:.3f}")
    print(f"Yield (>80% performance): {yield_80:.1f}%")
    print(f"Yield (>90% performance): {yield_90:.1f}%")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Performance distribution
    ax1 = axes[0, 0]
    bins = np.linspace(0.5, 1.0, 21)
    ax1.hist(performances, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(mean_performance, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_performance:.3f}')
    ax1.axvline(0.8, color='green', linestyle=':', linewidth=2, label='80% threshold')
    ax1.set_xlabel('Performance Relative to Ideal', fontsize=11)
    ax1.set_ylabel('Number of Devices', fontsize=11)
    ax1.set_title(f'Fabrication Yield Analysis (N={n_devices})', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative distribution
    ax2 = axes[0, 1]
    performance_sorted = np.sort(performances)
    cdf = np.arange(1, len(performance_sorted) + 1) / len(performance_sorted)
    ax2.plot(performance_sorted, cdf, 'b-', linewidth=3)
    ax2.axhline(yield_80/100, color='green', linestyle=':', alpha=0.7)
    ax2.axvline(0.8, color='green', linestyle=':', alpha=0.7)
    ax2.text(0.55, yield_80/100 + 0.02, f'{yield_80:.1f}% > 80%', fontsize=10, color='green')
    ax2.set_xlabel('Performance Threshold', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution Function', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Example fabricated devices
    ax3 = axes[1, 0]
    
    # Show 4 example devices
    example_indices = [0, 250, 500, 750]
    for idx, device_idx in enumerate(example_indices):
        # Recreate the same device (using seed)
        np.random.seed(device_idx)
        error_mask = np.random.normal(0, 1, grid_size)
        error_mask = gaussian_filter(error_mask, sigma=2)
        placement_error = error_mask * 10 / 1
        fabricated_example = np.where(final_design > 0.5 + placement_error/100, 1.0, 0.0)
        
        # Plot small version
        sub_ax = fig.add_subplot(4, 4, 13 + idx)  # Bottom left grid
        sub_ax.imshow(fabricated_example.T, cmap='binary', vmin=0, vmax=1)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(f'Device {device_idx}', fontsize=8)
    
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'Example Fabricated Devices', 
            transform=ax3.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    # Subplot 4: Error sensitivity analysis
    ax4 = axes[1, 1]
    
    # Test different error levels
    error_levels = [0, 5, 10, 15, 20]  # nm
    yields_at_levels = []
    
    for error_std in error_levels:
        # Quick simulation for this error level
        np.random.seed(42)
        n_test = 100
        test_performances = []
        
        for _ in range(n_test):
            error_mask = np.random.normal(0, 1, grid_size)
            error_mask = gaussian_filter(error_mask, sigma=2)
            placement_error = error_mask * error_std / 1
            fabricated = np.where(final_design > 0.5 + placement_error/100, 1.0, 0.0)
            
            ideal_gold = final_design > 0.7
            fabricated_gold = fabricated > 0.5
            if np.sum(ideal_gold) > 0:
                overlap = np.sum(ideal_gold & fabricated_gold)
                performance = overlap / np.sum(ideal_gold)
                test_performances.append(performance)
        
        yield_level = np.mean(np.array(test_performances) > 0.8) * 100
        yields_at_levels.append(yield_level)
    
    ax4.plot(error_levels, yields_at_levels, 'o-', linewidth=2, markersize=8, color='crimson')
    ax4.set_xlabel('Fabrication Error Standard Deviation (nm)', fontsize=11)
    ax4.set_ylabel('Yield (>80% performance) %', fontsize=11)
    ax4.set_title('Error Sensitivity Analysis', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Mark typical e-beam error (10 nm)
    ax4.axvline(10, color='blue', linestyle='--', alpha=0.7)
    ax4.text(10.5, 50, 'Typical e-beam\n(10 nm)', fontsize=9, color='blue')
    
    plt.suptitle('Experiment 1: Monte Carlo Fabrication Analysis', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/experiments/fig6a_monte_carlo_fabrication.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved: {output_path}")
    
    # Save data
    fabrication_data = {
        'n_devices': n_devices,
        'mean_performance': float(mean_performance),
        'std_performance': float(std_performance),
        'yield_80': float(yield_80),
        'yield_90': float(yield_90),
        'performances': performances.tolist(),
        'error_sensitivity': {
            'error_levels': error_levels,
            'yields': yields_at_levels
        }
    }
    
    np.savez("data/experimental_comparison/fabrication_analysis.npz", **fabrication_data)
    
    # Create LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Monte Carlo fabrication analysis results (N={n_devices})}}
\\label{{tab:fabrication_yield}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Mean Performance & {mean_performance:.3f} \\\\
Standard Deviation & {std_performance:.3f} \\\\
Yield (>80\\% performance) & {yield_80:.1f}\\% \\\\
Yield (>90\\% performance) & {yield_90:.1f}\\% \\\\
Devices below 70\\% & {np.mean(performances < 0.7)*100:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open("tables/fabrication_yield.tex", "w") as f:
        f.write(latex_table)
    
    print("✓ Data saved: data/experimental_comparison/fabrication_analysis.npz")
    print("✓ Table saved: tables/fabrication_yield.tex")
    
    return True

if __name__ == "__main__":
    monte_carlo_fabrication_analysis()
