#!/usr/bin/env python3
"""
Fix Time-Resolved: Realistic detection thresholds
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def fix_time_resolved():
    """Fix time-resolved detection with realistic thresholds"""
    print("Fixing Experiment 5: Realistic time-resolved sensing...")
    
    # Load trajectory data
    try:
        trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
        positions = trajectory_data['positions']
        density = trajectory_data['density']
    except FileNotFoundError:
        print("❌ Error: Trajectory data not found")
        return False
    
    # Load design
    final_design = np.load("data/optimization/final_design.npy")
    grid_size = final_design.shape
    
    # Simulate realistic SERS enhancement factors
    # Typical values: 10^6-10^8 for single molecules
    # We'll simulate log-normal distribution
    
    n_frames = min(1000, positions.shape[0])  # Use first 1000 frames
    
    print(f"Simulating {n_frames} time points with realistic SERS...")
    
    # REALISTIC PARAMETERS:
    base_enhancement = 1e6  # 10^6 typical for good SERS substrates
    enhancement_variation = 0.3  # 30% variation
    detection_threshold = 5e5   # Realistic detection limit
    noise_floor = 1e4           # Background signal
    
    # Time points
    time_points = np.arange(n_frames) * 0.01  # 10 ps per frame = 10 ns total
    
    # Generate realistic signals
    np.random.seed(42)
    
    # 1. Calculate field enhancement at each position
    signals = []
    detected = []
    
    for i in range(n_frames):
        # Get molecular position
        pos = positions[i]
        x, y = pos
        
        # Convert to integer indices
        ix = int(np.clip(x, 0, grid_size[0]-1))
        iy = int(np.clip(y, 0, grid_size[1]-1))
        
        # Field enhancement factor at this position
        # In real SERS: enhancement ∝ |E|^4
        design_value = final_design[ix, iy]
        
        # Realistic enhancement model
        if design_value > 0.7:  # In gold region
            # Hotspot: very high enhancement
            enhancement_factor = base_enhancement * (1 + enhancement_variation * np.random.randn())
            # |E|^4 scaling for SERS
            signal_strength = enhancement_factor * (design_value ** 4)
        elif design_value > 0.3:  # Near edge
            # Moderate enhancement
            enhancement_factor = base_enhancement * 0.1 * (1 + enhancement_variation * np.random.randn())
            signal_strength = enhancement_factor * (design_value ** 4)
        else:  # In water
            # Low background
            enhancement_factor = base_enhancement * 0.01
            signal_strength = enhancement_factor
        
        # Add noise
        signal_with_noise = signal_strength + noise_floor * np.random.exponential()
        
        signals.append(signal_with_noise)
        detected.append(signal_with_noise > detection_threshold)
    
    signals = np.array(signals)
    detected = np.array(detected)
    
    # Calculate statistics
    detection_fraction = np.mean(detected) * 100
    mean_signal = np.mean(signals)
    std_signal = np.std(signals)
    
    # Calculate correlation with design value at positions
    design_values = []
    for i in range(n_frames):
        pos = positions[i]
        x, y = pos
        ix = int(np.clip(x, 0, grid_size[0]-1))
        iy = int(np.clip(y, 0, grid_size[1]-1))
        design_values.append(final_design[ix, iy])
    
    design_values = np.array(design_values)
    correlation = np.corrcoef(design_values, signals)[0, 1]
    
    print(f"\n=== Realistic Time-Resolved Results ===")
    print(f"Detection fraction: {detection_fraction:.1f}%")
    print(f"Mean signal: {mean_signal:.2e}")
    print(f"Signal std: {std_signal:.2e}")
    print(f"Correlation with design: {correlation:.3f}")
    
    # Create fixed figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Time trace of signal
    ax1 = axes[0, 0]
    
    # Smooth signal for visualization
    from scipy.ndimage import gaussian_filter1d
    signal_smooth = gaussian_filter1d(signals, sigma=5)
    
    ax1.plot(time_points, signals, 'gray', alpha=0.3, linewidth=0.5, label='Raw')
    ax1.plot(time_points, signal_smooth, 'b-', linewidth=2, label='Smoothed')
    ax1.axhline(detection_threshold, color='red', linestyle='--', 
               linewidth=2, label=f'Threshold: {detection_threshold:.0e}')
    
    # Fill detected regions
    detected_regions = detected.astype(int)
    for i in range(len(detected_regions)-1):
        if detected_regions[i] == 1:
            ax1.axvspan(time_points[i], time_points[i+1], alpha=0.2, color='green')
    
    ax1.set_xlabel('Time (ns)', fontsize=11)
    ax1.set_ylabel('SERS Signal (a.u.)', fontsize=11)
    ax1.set_title('Time-Resolved SERS Signal', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Detection events
    ax2 = axes[0, 1]
    
    # Create histogram of signals
    log_signals = np.log10(signals[signals > 0])
    bins = np.linspace(4, 9, 21)  # 10^4 to 10^9
    
    ax2.hist(log_signals, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.log10(detection_threshold), color='red', linestyle='--',
               linewidth=2, label=f'Threshold: 10^{np.log10(detection_threshold):.1f}')
    
    # Add detected fraction annotation
    ax2.text(np.log10(detection_threshold) + 0.2, ax2.get_ylim()[1]*0.8,
            f'Detected: {detection_fraction:.1f}%', fontsize=10, color='red')
    
    ax2.set_xlabel('log₁₀(Signal)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Signal Distribution', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Correlation analysis
    ax3 = axes[1, 0]
    
    scatter = ax3.scatter(design_values, np.log10(signals), 
                         c=time_points, cmap='viridis', alpha=0.6, s=20)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(design_values, np.log10(signals))
    x_fit = np.linspace(0, 1, 100)
    y_fit = slope * x_fit + intercept
    ax3.plot(x_fit, y_fit, 'r--', linewidth=2, 
            label=f'r = {correlation:.3f}')
    
    ax3.set_xlabel('Design Value at Molecule Position', fontsize=11)
    ax3.set_ylabel('log₁₀(Signal)', fontsize=11)
    ax3.set_title(f'Signal-Design Correlation (r = {correlation:.3f})', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Time (ns)', fontsize=10)
    
    # Subplot 4: Detection map
    ax4 = axes[1, 1]
    
    # Create detection probability map
    detection_map = np.zeros(grid_size)
    count_map = np.zeros(grid_size)
    
    for i in range(n_frames):
        if detected[i]:
            pos = positions[i]
            x, y = pos
            ix = int(np.clip(x, 0, grid_size[0]-1))
            iy = int(np.clip(y, 0, grid_size[1]-1))
            detection_map[ix, iy] += 1
            count_map[ix, iy] += 1
        else:
            pos = positions[i]
            x, y = pos
            ix = int(np.clip(x, 0, grid_size[0]-1))
            iy = int(np.clip(y, 0, grid_size[1]-1))
            count_map[ix, iy] += 1
    
    # Calculate probability
    probability_map = np.zeros_like(detection_map)
    mask = count_map > 0
    probability_map[mask] = detection_map[mask] / count_map[mask]
    
    # Overlay on design
    im = ax4.imshow(final_design.T, cmap='gray_r', alpha=0.7, vmin=0, vmax=1)
    
    # Overlay detection probability
    prob_im = ax4.imshow(probability_map.T, cmap='hot', alpha=0.5, 
                        vmin=0, vmax=1, extent=[0, grid_size[0], 0, grid_size[1]])
    
    ax4.set_xlabel('x (nm)', fontsize=11)
    ax4.set_ylabel('y (nm)', fontsize=11)
    ax4.set_title('Detection Probability Map', fontsize=12)
    
    # Add colorbars
    cbar1 = plt.colorbar(im, ax=ax4, label='Design Value')
    cbar2 = plt.colorbar(prob_im, ax=ax4, label='Detection Probability')
    
    plt.suptitle('Fixed: Realistic Time-Resolved Single-Molecule Sensing', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save fixed figure
    output_path = "figures/experiments/fig6e_time_resolved_sensing_FIXED.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Fixed figure saved: {output_path}")
    
    # Save fixed data
    time_resolved_data = {
        'n_frames': n_frames,
        'detection_fraction': float(detection_fraction),
        'mean_signal': float(mean_signal),
        'std_signal': float(std_signal),
        'correlation': float(correlation),
        'detection_threshold': float(detection_threshold),
        'signals': signals.tolist(),
        'detected': detected.astype(int).tolist(),
        'time_points': time_points.tolist()
    }
    
    np.savez("data/experimental_comparison/time_resolved_FIXED.npz", **time_resolved_data)
    
    # Update LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Time-resolved single-molecule sensing performance}}
\\label{{tab:time_resolved}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Detection Fraction & {detection_fraction:.1f}\\% \\\\
Mean Signal & {mean_signal:.2e} \\\\
Signal Variation (CV) & {std_signal/mean_signal*100:.1f}\\% \\\\
Correlation with Design & {correlation:.3f} \\\\
Detection Threshold & {detection_threshold:.0e} \\\\
Frames Analyzed & {n_frames} \\\\
Time Window & {time_points[-1]:.1f} ns \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open("tables/time_resolved_sensing_FIXED.tex", "w") as f:
        f.write(latex_table)
    
    print("✓ Fixed data saved")
    print("✓ Updated table saved")
    
    return True

if __name__ == "__main__":
    fix_time_resolved()
