import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))

# Define positions in a GRID (no overlap guaranteed)
positions = {
    'Design': (4, 9),
    'Material': (4, 7),
    'FDTD': (2, 5),
    'MD': (6, 5),
    'E-field': (2, 3),
    'Trajectory': (6, 3),
    'Interpolation': (4, 3),
    'Signal': (4, 1),
    'Loss': (4, -1),
    'Gradient': (2, -3),
    'Update': (6, -3),
}

# Draw all nodes
for name, (x, y) in positions.items():
    ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue' if 'Design' in name else 
                    'orange' if 'FDTD' in name else
                    'lightgreen' if 'MD' in name else
                    'lightyellow' if 'Signal' in name else
                    'lightcoral' if 'Loss' in name else 'white'))

# Draw ONLY HORIZONTAL or VERTICAL arrows (no diagonal = no overlap)
# Vertical arrows
for pair in [('Design', 'Material'), ('Material', 'FDTD'), ('Material', 'MD'),
             ('FDTD', 'E-field'), ('MD', 'Trajectory'),
             ('E-field', 'Interpolation'), ('Trajectory', 'Interpolation'),
             ('Interpolation', 'Signal'), ('Signal', 'Loss'),
             ('Loss', 'Gradient'), ('Gradient', 'Update'), ('Update', 'Design')]:
    x1, y1 = positions[pair[0]]
    x2, y2 = positions[pair[1]]
    
    # Only draw if directly above/below or left/right
    if abs(x1 - x2) < 0.1:  # Vertical
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    elif abs(y1 - y2) < 0.1:  # Horizontal
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax.text(4, 8, "↓ Forward", ha='center', color='blue', fontweight='bold')
ax.text(4, -2, "↑ Backward", ha='center', color='red', fontweight='bold')

ax.set_xlim(0, 8)
ax.set_ylim(-4, 10)
ax.axis('off')
ax.set_title("Figure 1a: Computational Graph", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/fig1a_computational_graph.png", dpi=600)
print("✅ Created CLEAN Figure 1a with NO overlap")
