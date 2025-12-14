#!/usr/bin/env python3
"""
Generate Figure S5: Optimization convergence
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def create_figure_S5():
    """Create Figure S5: Optimization convergence"""
    print("Creating Figure S5: Optimization convergence...")
    
    # Load optimization data
    try:
        loss_history = np.load("data/optimization/loss_history.npy")
        gradient_history = np.load("data/optimization/gradient_history.npy")
        performance_data = np.load("data/optimization/performance_data.npz")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run optimization first (scripts/phase4/05_main_optimization.py)")
        return False
    
    epochs = np.arange(len(loss_history))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Loss convergence
    ax1.plot(epochs, loss_history, 'b-', linewidth=2, label='Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (Normalized)', fontsize=11)
    ax1.set_title('Loss Convergence', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add convergence annotation
    convergence_epoch = performance_data['convergence_epoch']
    if convergence_epoch < len(loss_history):
        ax1.axvline(convergence_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(convergence_epoch + 5, np.mean(loss_history),
                f'Convergence: epoch {convergence_epoch}', 
                fontsize=9, color='red', va='center')
    
    # Mark improvement
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    improvement = (initial_loss - final_loss) / abs(initial_loss) * 100
    
    ax1.text(len(loss_history)*0.7, loss_history[0]*0.9,
            f'Improvement: {improvement:.1f}%', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bottom: Gradient norm
    ax2.plot(epochs, gradient_history, 'r-', linewidth=2, label='Gradient Norm')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Gradient Norm', fontsize=11)
    ax2.set_title('Gradient Norm Evolution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')  # Log scale for gradient
    
    # Add final gradient annotation
    final_gradient = gradient_history[-1]
    ax2.text(len(gradient_history)*0.7, gradient_history[0]*0.1,
            f'Final gradient: {final_gradient:.2e}', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.suptitle('Supplementary Figure S5: Optimization Convergence Analysis', 
                 fontsize=14, y=0.98)
    
    # Fix tight layout warning by adjusting spacing
    plt.subplots_adjust(hspace=0.3, top=0.93)
    
    # Save figure
    output_path = "figures/supplementary/figS5_convergence.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure S5 saved to: {output_path}")
    
    # Create convergence metrics
    convergence_metrics = {
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'improvement_percent': float(improvement),
        'convergence_epoch': int(convergence_epoch),
        'final_gradient': float(final_gradient),
        'total_epochs': len(loss_history)
    }
    
    # Save metrics
    np.save("data/optimization/analysis/convergence_metrics.npy", convergence_metrics)
    
    # SIMPLIFIED LaTeX table - no formatting issues
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Optimization convergence metrics}}
\\label{{tab:convergence_metrics}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Initial Loss & {convergence_metrics['initial_loss']:.4f} \\\\
Final Loss & {convergence_metrics['final_loss']:.4f} \\\\
Improvement & {convergence_metrics['improvement_percent']:.1f}\\% \\\\
Convergence Epoch & {convergence_metrics['convergence_epoch']} \\\\
Final Gradient Norm & {convergence_metrics['final_gradient']:.2e} \\\\
Total Epochs & {convergence_metrics['total_epochs']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open("tables/convergence_metrics.tex", "w") as f:
        f.write(latex_table)
    
    print("✓ Convergence metrics saved to tables/convergence_metrics.tex")
    print("✅ Figure S5 generation complete!")
    
    return True

if __name__ == "__main__":
    create_figure_S5()
