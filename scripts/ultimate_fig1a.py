# Save as: scripts/ultimate_fig1a.py
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

# Simple linear pipeline (no overlap possible)
stages = ["Design θ", "FDTD + MD", "Signal S", "Loss ℒ", "Gradient ∇ℒ", "Update"]

for i, stage in enumerate(stages):
    x = i * 2
    ax.text(x, 0, stage, ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=['lightblue', 'orange', 'lightgreen', 
                                                   'lightcoral', 'violet', 'lightblue'][i]))

# Simple right arrows
for i in range(len(stages)-1):
    ax.annotate('', xy=((i+1)*2, 0), xytext=(i*2, 0),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Loop back arrow (dashed)
ax.annotate('', xy=(0, -0.5), xytext=(10, -0.5),
           arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='dashed'))

ax.text(5, 0.3, "Forward", color='blue', fontweight='bold', ha='center')
ax.text(5, -0.7, "Backward", color='red', fontweight='bold', ha='center')

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 1)
ax.axis('off')
ax.set_title("Figure 1a: Differentiable Pipeline", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/fig1a_computational_graph.png", dpi=600)
print("✅ Created ULTRA-SIMPLE Figure 1a")
