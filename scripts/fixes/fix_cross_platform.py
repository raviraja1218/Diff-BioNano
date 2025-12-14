#!/usr/bin/env python3
"""
Fix Cross-Platform: Realistic fabrication method errors
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def fix_cross_platform():
    """Fix cross-platform fabrication with realistic errors"""
    print("Fixing Experiment 3: Realistic cross-platform fabrication...")
    
    # Load final design
    final_design = np.load("data/optimization/final_design.npy")
    grid_size = final_design.shape
    
    # Fabrication methods with REALISTIC parameters
    fabrication_methods = [
        {
            "name": "Electron-Beam Lithography",
            "abbrev": "E-beam",
            "placement_error": 8,   # nm
            "linewidth_error": 3,   # nm  
            "min_feature": 20,      # nm
            "edge_roughness": 2,    # nm
            "defect_rate": 0.001,
            "color": "royalblue"
        },
        {
            "name": "Nanoimprint Lithography", 
            "abbrev": "Nanoimprint",
            "placement_error": 15,  # nm
            "linewidth_error": 8,   # nm
            "min_feature": 50,      # nm
            "edge_roughness": 5,    # nm
            "defect_rate": 0.0005,
            "color": "forestgreen"
        },
        {
            "name": "DNA Origami Assembly",
            "abbrev": "DNA Origami",
            "placement_error": 2,   # nm (very precise!)
            "linewidth_error": 1,   # nm
            "min_feature": 5,       # nm
            "edge_roughness": 1,    # nm
            "defect_rate": 0.01,    # higher defect rate
            "max_size": 100,        # limited size
            "color": "darkorange"
        }
    ]
    
    performances = {}
    n_simulations = 500  # Per method
    
    print("Simulating fabrication methods with realistic errors...")
    
    for method in fabrication_methods:
        print(f"  {method['name']}...")
        method_perfs = []
        
        for i in range(n_simulations):
            # Apply method-specific errors
            
            # 1. Placement error (Gaussian)
            placement_noise = np.random.normal(0, method['placement_error']/2, grid_size)
            placement_noise = gaussian_filter(placement_noise, sigma=1.0)
            
            # 2. Linewidth variation
            linewidth_noise = np.random.normal(0, method['linewidth_error']/2, grid_size)
            linewidth_noise = gaussian_filter(linewidth_noise, sigma=0.8)
            
            # 3. Combine errors
            total_noise = placement_noise * 0.04 + linewidth_noise * 0.03
            
            # 4. Adjust threshold
            threshold = 0.5 + total_noise
            
            # 5. Create fabricated design
            fabricated = np.where(final_design > threshold, 1.0, 0.0)
            
            # 6. Apply edge roughness (mild)
            if method['edge_roughness'] > 0:
                roughness = np.random.normal(0, method['edge_roughness']/3, grid_size)
                roughness = gaussian_filter(roughness, sigma=0.5)
                fabricated = np.clip(fabricated + roughness * 0.05, 0, 1)
            
            # 7. Remove features below minimum size (simplified)
            min_feature_pixels = method['min_feature']  # 1 nm = 1 pixel
            if min_feature_pixels > 1:
                from scipy.ndimage import label
                labeled, num_features = label(fabricated > 0.5)
                for j in range(1, num_features + 1):
                    feature_mask = labeled == j
                    if np.sum(feature_mask) < min_feature_pixels**2:
                        fabricated[feature_mask] = 0.0
            
            # 8. Add defects
            defect_mask = np.random.random(grid_size) < method['defect_rate']
            fabricated[defect_mask] = 1 - fabricated[defect_mask]  # Flip
            
            # 9. For DNA origami: limit size
            if 'max_size' in method and method['max_size'] < grid_size[0]:
                # Center crop
                start = (grid_size[0] - method['max_size']) // 2
                end = start + method['max_size']
                region = fabricated[start:end, start:end]
                # Pad back to original size
                fabricated = np.zeros(grid_size)
                fabricated[start:end, start:end] = region
            
            # Calculate performance
            ideal_binary = final_design > 0.7
            fab_binary = fabricated > 0.5
            
            if np.sum(ideal_binary) > 0:
                overlap = np.sum(ideal_binary & fab_binary)
                performance = overlap / np.sum(ideal_binary)
                
                # Penalize extra material
                extra = np.sum(fab_binary & ~ideal_binary)
                performance -= 0.05 * extra / np.sum(ideal_binary)
                performance = max(0, min(1, performance))
            else:
                performance = 0.0
            
            method_perfs.append(performance)
        
        performances[method['name']] = {
            'performances': np.array(method_perfs),
            'mean': np.mean(method_perfs),
            'std': np.std(method_perfs),
            'yield_80': np.mean(np.array(method_perfs) > 0.8) * 100,
            'color': method['color']
        }
    
    # Create fixed figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Performance comparison
    ax1 = axes[0, 0]
    method_names = [m['name'] for m in fabrication_methods]
    mean_perfs = [performances[name]['mean'] for name in method_names]
    std_perfs = [performances[name]['std'] for name in method_names]
    colors = [performances[name]['color'] for name in method_names]
    
    bars = ax1.bar(range(len(method_names)), mean_perfs, yerr=std_perfs,
                  color=colors, edgecolor='black', linewidth=1.5,
                  error_kw=dict(ecolor='black', lw=1, capsize=5))
    
    ax1.set_xticks(range(len(method_names)))
    ax1.set_xticklabels([m['abbrev'] for m in fabrication_methods], fontsize=10)
    ax1.set_ylabel('Mean Performance', fontsize=11)
    ax1.set_title('Cross-Platform Fabrication Performance', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, mean_perfs)):
        ax1.text(bar.get_x() + bar.get_width()/2, mean_val + 0.03,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    # Subplot 2: Yield comparison
    ax2 = axes[0, 1]
    yields_80 = [performances[name]['yield_80'] for name in method_names]
    
    bars2 = ax2.bar(range(len(method_names)), yields_80, color=colors,
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels([m['abbrev'] for m in fabrication_methods], fontsize=10)
    ax2.set_ylabel('Yield (>80% performance) %', fontsize=11)
    ax2.set_title('Fabrication Yield by Method', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    for i, (bar, yield_val) in enumerate(zip(bars2, yields_80)):
        ax2.text(bar.get_x() + bar.get_width()/2, yield_val + 2,
                f'{yield_val:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    # Subplot 3: Error parameter visualization
    ax3 = axes[1, 0]
    
    # Error parameters table
    params = ['Placement Error', 'Linewidth Error', 'Min Feature', 'Edge Roughness']
    ebeam_params = [fabrication_methods[0]['placement_error'], 
                   fabrication_methods[0]['linewidth_error'],
                   fabrication_methods[0]['min_feature'],
                   fabrication_methods[0]['edge_roughness']]
    
    nanoimprint_params = [fabrication_methods[1]['placement_error'],
                         fabrication_methods[1]['linewidth_error'],
                         fabrication_methods[1]['min_feature'],
                         fabrication_methods[1]['edge_roughness']]
    
    dna_params = [fabrication_methods[2]['placement_error'],
                 fabrication_methods[2]['linewidth_error'],
                 fabrication_methods[2]['min_feature'],
                 fabrication_methods[2]['edge_roughness']]
    
    x = np.arange(len(params))
    width = 0.25
    
    bars3a = ax3.bar(x - width, ebeam_params, width, label='E-beam', color='royalblue')
    bars3b = ax3.bar(x, nanoimprint_params, width, label='Nanoimprint', color='forestgreen')
    bars3c = ax3.bar(x + width, dna_params, width, label='DNA Origami', color='darkorange')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(params, fontsize=9)
    ax3.set_ylabel('Error Magnitude (nm)', fontsize=11)
    ax3.set_title('Fabrication Method Parameters', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Example fabricated structures
    ax4 = axes[1, 1]
    
    # Show examples for each method
    for idx, method in enumerate(fabrication_methods[:3]):
        np.random.seed(100 + idx)
        
        placement_noise = np.random.normal(0, method['placement_error']/2, grid_size)
        placement_noise = gaussian_filter(placement_noise, sigma=1.0)
        linewidth_noise = np.random.normal(0, method['linewidth_error']/2, grid_size)
        linewidth_noise = gaussian_filter(linewidth_noise, sigma=0.8)
        
        total_noise = placement_noise * 0.04 + linewidth_noise * 0.03
        threshold = 0.5 + total_noise
        example = np.where(final_design > threshold, 1.0, 0.0)
        
        # Show small region
        region_size = 50
        start_x = 75
        start_y = 75
        
        region = example[start_x:start_x+region_size, start_y:start_y+region_size]
        
        sub_ax = fig.add_subplot(3, 3, 19 + idx)  # Bottom right grid
        sub_ax.imshow(region.T, cmap='binary', vmin=0, vmax=1)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        sub_ax.set_title(method['abbrev'], fontsize=8)
    
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'Example Fabricated Regions (50×50 nm)', 
            transform=ax4.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Fixed: Cross-Platform Fabrication Compatibility', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save fixed figure
    output_path = "figures/experiments/fig6c_cross_platform_fabrication_FIXED.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Fixed Cross-Platform Results ===")
    for method in fabrication_methods:
        name = method['name']
        perf = performances[name]
        print(f"{method['abbrev']}: Mean={perf['mean']:.3f}, Yield(80%)={perf['yield_80']:.1f}%")
    
    print(f"\n✓ Fixed figure saved: {output_path}")
    
    # Save fixed data
    cross_platform_data = {
        'methods': fabrication_methods,
        'performances': performances,
        'n_simulations': n_simulations
    }
    
    np.savez("data/experimental_comparison/cross_platform_FIXED.npz", **cross_platform_data)
    
    # Update LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Cross-platform fabrication compatibility}
\label{tab:cross_platform}
\begin{tabular}{lcccc}
\toprule
Method & Mean Performance & Yield (>80\%) & Min Feature (nm) & Placement Error (nm) \\
\midrule
"""
    
    for method in fabrication_methods:
        name = method['name']
        perf = performances[name]
        latex_table += f"{method['abbrev']} & {perf['mean']:.3f} & {perf['yield_80']:.1f}\% & {method['min_feature']} & {method['placement_error']} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open("tables/cross_platform_fabrication_FIXED.tex", "w") as f:
        f.write(latex_table)
    
    print("✓ Fixed data saved")
    print("✓ Updated table saved")
    
    return True

if __name__ == "__main__":
    fix_cross_platform()
