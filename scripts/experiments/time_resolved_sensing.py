#!/usr/bin/env python3
"""
Experiment 5: Time-resolved dynamic sensing simulation
Simulate real-time tracking of molecular motion
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def time_resolved_sensing():
    """Simulate real-time tracking of molecular dynamics"""
    print("Running Experiment 5: Time-resolved dynamic sensing...")
    
    # Load trajectory data
    trajectory_data = np.load("data/optimization/trajectory_loaded.npz")
    positions = trajectory_data['positions']
    
    # Use first 100 time points for visualization
    n_frames = min(100, positions.shape[0])
    times = np.arange(n_frames) * 0.01  # 10 ps per frame
    
    # Simulate sensor response
    # Sensor hotspots track molecular positions with some delay/response time
    np.random.seed(42)
    
    # Create simulated sensor hotspots (from our design)
    # Assume 3 main hotspots in the optimized design
    hotspots = np.array([
        [80, 120],   # Hotspot 1
        [120, 80],   # Hotspot 2  
        [100, 100],  # Hotspot 3
    ])
    
    # Calculate distances to nearest hotspot over time
    distances = []
    sensor_responses = []
    
    for t in range(n_frames):
        pos = positions[t]
        # Find distance to nearest hotspot
        min_dist = float('inf')
        for hotspot in hotspots:
            dist = np.sqrt(np.sum((pos - hotspot)**2))
            if dist < min_dist:
                min_dist = dist
        
        distances.append(min_dist)
        
        # Sensor response: stronger when molecule is closer
        # Response = exp(-distance^2 / (2*sigma^2)), sigma = 20 pixels
        response = np.exp(-min_dist**2 / (2 * 20**2))
        sensor_responses.append(response)
    
    # Add some sensor response dynamics (low-pass filter)
    # Real sensors have finite response time
    b, a = signal.butter(2, 0.1)  # Low-pass filter, cutoff at 0.1 * Nyquist
    filtered_response = signal.filtfilt(b, a, sensor_responses)
    
    # Add noise (typical for experimental measurements)
    measurement_noise = np.random.normal(0, 0.05, n_frames)
    noisy_response = filtered_response + measurement_noise
    noisy_response = np.clip(noisy_response, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Molecular trajectory with hotspots
    ax1 = axes[0, 0]
    
    # Plot trajectory
    ax1.plot(positions[:n_frames, 0], positions[:n_frames, 1], 
            'b-', alpha=0.5, linewidth=1, label='Molecular trajectory')
    ax1.scatter(positions[:n_frames, 0], positions[:n_frames, 1], 
               c=times, cmap='viridis', s=20, alpha=0.7, label='Time points')
    
    # Plot hotspots
    ax1.scatter(hotspots[:, 0], hotspots[:, 1], 
               s=200, c='red', marker='*', edgecolors='black', linewidth=2,
               label='Sensor hotspots', zorder=5)
    
    # Add labels for hotspots
    for i, hotspot in enumerate(hotspots):
        ax1.text(hotspot[0] + 5, hotspot[1] + 5, f'HS{i+1}', 
                fontsize=10, fontweight='bold', color='red')
    
    ax1.set_xlabel('x position (pixels)', fontsize=11)
    ax1.set_ylabel('y position (pixels)', fontsize=11)
    ax1.set_title('Molecular Trajectory and Sensor Hotspots', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Distance to nearest hotspot over time
    ax2 = axes[0, 1]
    
    ax2.plot(times, distances, 'r-', linewidth=2, label='Distance to nearest hotspot')
    ax2.fill_between(times, 0, distances, alpha=0.3, color='red')
    
    # Mark when molecule is close (< 20 pixels)
    close_mask = np.array(distances) < 20
    ax2.scatter(times[close_mask], np.array(distances)[close_mask], 
               color='green', s=50, zorder=5, label='Close to hotspot (<20 px)')
    
    ax2.set_xlabel('Time (ps)', fontsize=11)
    ax2.set_ylabel('Distance to nearest hotspot (pixels)', fontsize=11)
    ax2.set_title('Distance Tracking Over Time', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(distances) * 1.1)
    
    # 3. Sensor response over time
    ax3 = axes[1, 0]
    
    ax3.plot(times, sensor_responses, 'b-', alpha=0.5, linewidth=1, label='Ideal response')
    ax3.plot(times, filtered_response, 'g-', linewidth=2, label='Filtered (sensor dynamics)')
    ax3.plot(times, noisy_response, 'r-', linewidth=2, label='Measured (with noise)')
    
    ax3.set_xlabel('Time (ps)', fontsize=11)
    ax3.set_ylabel('Normalized Sensor Response', fontsize=11)
    ax3.set_title('Real-Time Sensor Response', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Mark response peaks
    peaks, _ = signal.find_peaks(noisy_response, height=0.5, distance=10)
    ax3.scatter(times[peaks], noisy_response[peaks], 
               color='purple', s=100, zorder=5, edgecolors='black', linewidth=2,
               label='Detection events')
    
    # 4. Correlation analysis
    ax4 = axes[1, 1]
    
    # Calculate correlation between distance and response
    # When distance is small, response should be high (negative correlation)
    correlation = np.corrcoef(distances, -noisy_response)[0, 1]
    
    # Scatter plot
    scatter = ax4.scatter(distances, noisy_response, 
                         c=times, cmap='viridis', s=50, alpha=0.7)
    
    # Fit line
    z = np.polyfit(distances, noisy_response, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min(distances), max(distances), 100)
    ax4.plot(x_fit, p(x_fit), 'r--', linewidth=2, 
            label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    ax4.set_xlabel('Distance to hotspot (pixels)', fontsize=11)
    ax4.set_ylabel('Sensor Response', fontsize=11)
    ax4.set_title(f'Distance-Response Correlation (r = {correlation:.3f})', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for time
    plt.colorbar(scatter, ax=ax4, label='Time (ps)')
    
    plt.suptitle('Experiment 5: Time-Resolved Dynamic Sensing Simulation', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/experiments/fig6e_time_resolved_sensing.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved: {output_path}")
    
    # Calculate detection metrics
    detection_threshold = 0.5
    detections = noisy_response > detection_threshold
    n_detections = np.sum(detections)
    detection_fraction = n_detections / n_frames * 100
    
    # Time resolution: minimum time between detectable changes
    response_changes = np.abs(np.diff(noisy_response)) > 0.1
    if np.sum(response_changes) > 0:
        change_times = times[1:][response_changes]
        if len(change_times) > 1:
            time_resolution = np.min(np.diff(change_times))
        else:
            time_resolution = times[-1] / n_frames
    else:
        time_resolution = times[-1] / n_frames
    
    # Save data
    time_data = {
        "times": times.tolist(),
        "distances": distances,
        "sensor_responses": sensor_responses,
        "filtered_response": filtered_response.tolist(),
        "noisy_response": noisy_response.tolist(),
        "correlation": float(correlation),
        "detection_fraction": float(detection_fraction),
        "time_resolution": float(time_resolution),
        "n_detections": int(n_detections),
        "peaks": peaks.tolist()
    }
    
    np.savez("data/experimental_comparison/time_resolved_sensing.npz", **time_data)
    
    # Create LaTeX table
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Time-resolved sensing performance metrics}}
\\label{{tab:time_resolved}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Correlation (distance-response) & {correlation:.3f} \\\\
Detection fraction (threshold=0.5) & {detection_fraction:.1f}\\% \\\\
Number of detection events & {n_detections} \\\\
Estimated time resolution & {time_resolution:.2f} ps \\\\
Total simulation time & {times[-1]:.1f} ps \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open("tables/time_resolved_sensing.tex", "w") as f:
        f.write(latex_table)
    
    print("\n=== Time-Resolved Sensing Results ===")
    print(f"Correlation between distance and response: {correlation:.3f}")
    print(f"Detection fraction: {detection_fraction:.1f}% of frames")
    print(f"Time resolution: {time_resolution:.2f} ps")
    print(f"Number of detection events: {n_detections}")
    
    print("✓ Data saved: data/experimental_comparison/time_resolved_sensing.npz")
    print("✓ Table saved: tables/time_resolved_sensing.tex")
    
    return True

if __name__ == "__main__":
    time_resolved_sensing()
