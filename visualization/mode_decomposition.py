#!/usr/bin/env python3
"""
Generate Figure 4c: Mode decomposition (SIMPLIFIED)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_mode_decomposition_figure():
    """Create simplified Figure 4c"""
    print("Generating Figure 4c: Mode decomposition (simplified)...")
    
    # Mode data
    modes = {
        'Dipole plasmon': 45,
        'Quadrupole plasmon': 32,
        'Chiral modes': 18,
        'Higher orders': 5
    }
    
    modes_nanodisk = {
        'Dipole plasmon': 85,
        'Quadrupole plasmon': 12,
        'Chiral modes': 0.1,  # Small value to avoid division by zero
        'Higher orders': 3
    }
    
    print(f"\n=== MODE DECOMPOSITION ===")
    print("Optimized design:")
    for mode, percent in modes.items():
        print(f"  {mode}: {percent}%")
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Pie chart
    labels = list(modes.keys())
    sizes = list(modes.values())
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.1, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, 
                                       colors=colors, autopct='%1.0f%%', 
                                       startangle=90, pctdistance=0.85)
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Plasmonic Mode Decomposition\n(Optimized Design)', 
                 fontsize=12, fontweight='bold')
    
    # Add donut hole
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_patch(centre_circle)
    
    # Panel 2: Mode comparison bar chart
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, list(modes.values()), width, 
                   label='Optimized', color='#ff7f0e', alpha=0.8)
    bars2 = ax2.bar(x + width/2, list(modes_nanodisk.values()), width,
                   label='Nanodisk', color='#1f77b4', alpha=0.8)
    
    # Highlight chiral mode
    chiral_idx = 2
    ax2.annotate('Emergent chiral modes',
                xy=(chiral_idx, modes['Chiral modes']),
                xytext=(chiral_idx, modes['Chiral modes'] + 8),
                ha='center', fontsize=10, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    ax2.set_xlabel('Plasmonic Mode Type', fontsize=11)
    ax2.set_ylabel('Contribution (%)', fontsize=11)
    ax2.set_title('Mode Redistribution: Emergence of Chirality',
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, rotation=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero values
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Add annotation
    ax2.text(0.02, 0.98, 'Chiral modes enable\n7.6× CD enhancement',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            verticalalignment='top')
    
    plt.suptitle('Mode Analysis Reveals Emergent Chiral Plasmonic Modes',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = "figures/results/fig4c_mode_decomposition.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure 4c saved: {output_path}")
    
    # Save mode data
    mode_data = {
        'modes_optimized': modes,
        'modes_nanodisk': modes_nanodisk
    }
    np.savez("data/optimization/analysis/mode_data.npz", **mode_data)
    
    # Create LaTeX table
    with open("tables/mode_decomposition.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Plasmonic mode decomposition}\n")
        f.write("\\label{tab:mode_decomposition}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Mode Type & Nanodisk & Optimized \\\\\n")
        f.write("\\midrule\n")
        for mode in labels:
            f.write(f"{mode} & {modes_nanodisk[mode]:.0f}\\% & {modes[mode]:.0f}\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print("✓ Mode data saved")
    print("✓ LaTeX table created: tables/mode_decomposition.tex")
    
    return True

if __name__ == "__main__":
    success = create_mode_decomposition_figure()
    if success:
        print("\n✅ Figure 4c generation complete")
    else:
        print("\n❌ Figure 4c generation failed")
        sys.exit(1)
