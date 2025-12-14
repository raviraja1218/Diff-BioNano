#!/usr/bin/env python3
"""
Experiment 3: Cross-platform fabrication compatibility
Simulate same design via different fabrication methods - SIMPLIFIED VERSION
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def cross_platform_fabrication():
    """Simulate different fabrication methods - SIMPLIFIED"""
    print("Running Experiment 3: Cross-platform fabrication compatibility...")
    
    # Load final design
    final_design = np.load("data/optimization/final_design.npy")
    ideal_binary = final_design > 0.7
    
    # Define fabrication methods and their typical parameters
    fabrication_methods = [
        {
            "name": "E-beam Lithography",
            "min_feature": 20,  # nm
            "placement_error": 10,  # nm
            "linewidth_error": 5,   # nm
            "edge_roughness": 3,    # nm
            "cost": "High",
            "throughput": "Low"
        },
        {
            "name": "Nanoimprint Lithography",
            "min_feature": 50,
            "placement_error": 15,
            "linewidth_error": 10,
            "edge_roughness": 8,
            "cost": "Medium",
            "throughput": "High"
        },
        {
            "name": "DNA Origami Assembly",
            "min_feature": 5,
            "placement_error": 2,
            "linewidth_error": 1,
            "edge_roughness": 1,
            "cost": "Very High",
            "throughput": "Very Low"
        }
    ]
    
    # Simulate each method
    results = []
    simulated_designs = []
    
    for method in fabrication_methods:
        print(f"  Simulating {method['name']}...")
        
        # Apply method-specific errors
        np.random.seed(42)  # For reproducibility
        
        # Start with ideal
        fabricated = ideal_binary.copy()
        
        # 1. Feature size variation (dilation/erosion)
        size_variation = np.random.normal(0, method['linewidth_error']/2)
        if size_variation > 0:
            se_size = max(1, int(abs(size_variation)))
            structuring_element = np.ones((se_size, se_size))
            fabricated = binary_dilation(fabricated, structure=structuring_element)
        elif size_variation < 0:
            se_size = max(1, int(abs(size_variation)))
            structuring_element = np.ones((se_size, se_size))
            fabricated = binary_erosion(fabricated, structure=structuring_element)
        
        # 2. Placement errors
        shift_x = int(np.random.normal(0, method['placement_error']/2))
        shift_y = int(np.random.normal(0, method['placement_error']/2))
        if shift_x != 0 or shift_y != 0:
            fabricated = np.roll(fabricated, shift=(shift_x, shift_y), axis=(0, 1))
        
        # 3. Edge roughness
        if method['edge_roughness'] > 0:
            roughness = np.random.normal(0, 1, fabricated.shape)
            roughness = gaussian_filter(roughness, sigma=1)
            roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())
            roughness = roughness * 2 - 1  # Scale to [-1, 1]
            
            # Apply roughness as probabilistic erosion/dilation
            roughness_mask = roughness > np.random.uniform(-0.5, 0.5, fabricated.shape)
            fabricated = fabricated & roughness_mask
        
        # Convert to float for visualization
        fabricated_design = fabricated.astype(float)
        
        # Calculate performance
        if np.sum(ideal_binary) > 0:
            overlap = np.sum(ideal_binary & fabricated)
            performance = overlap / np.sum(ideal_binary)
        else:
            performance = 0.0
        
        # Store results
        results.append({
            "method": method['name'],
            "performance": performance,
            "min_feature": method['min_feature'],
            "placement_error": method['placement_error'],
            "cost": method['cost'],
            "throughput": method['throughput']
        })
        
        simulated_designs.append(fabricated_design)
    
    # Create figure - SIMPLIFIED LAYOUT
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Performance comparison (top left)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    methods = [r["method"] for r in results]
    performances = [r["performance"] for r in results]
    
    bars = ax1.bar(range(len(methods)), performances, 
                   color=['steelblue', 'forestgreen', 'goldenrod'])
    
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.split()[0] for m in methods], fontsize=10)
    ax1.set_ylabel('Performance Relative to Ideal', fontsize=11)
    ax1.set_title('Performance Across Fabrication Methods', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, perf in zip(bars, performances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{perf:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Minimum feature size vs performance (top right)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    min_features = [r["min_feature"] for r in results]
    
    scatter = ax2.scatter(min_features, performances, s=200, 
                         c=['steelblue', 'forestgreen', 'goldenrod'],
                         edgecolors='black', linewidth=2)
    
    # Add trend line
    z = np.polyfit(min_features, performances, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(min_features), max(min_features), 100)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Minimum Feature Size (nm)', fontsize=11)
    ax2.set_ylabel('Performance', fontsize=11)
    ax2.set_title('Performance vs Feature Size Limit', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Label points
    for i, (method, x, y) in enumerate(zip(methods, min_features, performances)):
        ax2.text(x + 2, y + 0.02, method.split()[0], fontsize=9)
    
    # Plot 3: Fabricated designs (bottom row)
    for idx, (design, method, perf) in enumerate(zip(simulated_designs, methods, performances)):
        ax = plt.subplot2grid((3, 4), (1 + idx//2, idx%2 * 2), colspan=2, rowspan=1)
        im = ax.imshow(design.T, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f'{method}\nPerformance: {perf:.3f}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add specs
        specs = f"Min feature: {results[idx]['min_feature']} nm\n"
        specs += f"Placement error: ±{results[idx]['placement_error']} nm\n"
        specs += f"Cost: {results[idx]['cost']}, Throughput: {results[idx]['throughput']}"
        
        ax.text(0.02, 0.98, specs, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Experiment 3: Cross-Platform Fabrication Compatibility', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/experiments/fig6c_cross_platform_fabrication.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved: {output_path}")
    
    # Save data
    fabrication_data = {
        "methods": fabrication_methods,
        "results": results,
        "simulated_designs": [d.tolist() for d in simulated_designs]
    }
    
    np.savez("data/experimental_comparison/cross_platform_fabrication.npz", **fabrication_data)
    
    # Create LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Cross-platform fabrication compatibility}
\label{tab:cross_platform}
\begin{tabular}{lcccccc}
\toprule
Method & Min Feature (nm) & Placement Error (nm) & Performance & Cost & Throughput & Recommended \\
\midrule
"""
    
    for i, r in enumerate(results):
        recommended = "✓" if r["performance"] > 0.8 else "–"
        latex_table += f"{r['method']} & {r['min_feature']} & {r['placement_error']} & {r['performance']:.3f} & {r['cost']} & {r['throughput']} & {recommended} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open("tables/cross_platform_fabrication.tex", "w") as f:
        f.write(latex_table)
    
    print("\n=== Cross-Platform Results ===")
    for r in results:
        print(f"{r['method']}: {r['performance']:.3f} performance")
    
    print("✓ Data saved: data/experimental_comparison/cross_platform_fabrication.npz")
    print("✓ Table saved: tables/cross_platform_fabrication.tex")
    
    return True

if __name__ == "__main__":
    cross_platform_fabrication()
