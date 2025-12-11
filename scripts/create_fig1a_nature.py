import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(14, 3))
ax = plt.gca()
ax.axis('off')

# Style helper
def box(x, y, w, h, text):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25",
        linewidth=1,
        edgecolor="#333333",
        facecolor="#E8E8E8"
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", linewidth=1))

# Draw boxes
box(0.05, 0.4, 0.15, 0.18, "Design\nε(x,y)")
box(0.25, 0.4, 0.15, 0.18, "Material\nModel")
box(0.45, 0.4, 0.15, 0.18, "FDTD\nSolver")
box(0.65, 0.4, 0.15, 0.18, "Field\nE(x,y,t)")
box(0.65, 0.1, 0.15, 0.18, "Trajectory\nr(t)")
box(0.45, 0.1, 0.15, 0.18, "MD\nSimulation")
box(0.85, 0.4, 0.15, 0.18, "Signal\n∫|E(r)|²dt")
box(1.05, 0.4, 0.15, 0.18, "Loss\nℒ")
box(1.25, 0.4, 0.15, 0.18, "Gradient\n∂ℒ/∂ε")
box(1.45, 0.4, 0.15, 0.18, "Optimizer\nAdam")

# Arrows
arrow(0.20, 0.49, 0.25, 0.49)
arrow(0.40, 0.49, 0.45, 0.49)
arrow(0.60, 0.49, 0.65, 0.49)
arrow(0.80, 0.49, 0.85, 0.49)
arrow(1.00, 0.49, 1.05, 0.49)
arrow(1.20, 0.49, 1.25, 0.49)
arrow(1.40, 0.49, 1.45, 0.49)

# MD Path
arrow(0.52, 0.40, 0.52, 0.28)
arrow(0.52, 0.20, 0.52, 0.10)
arrow(0.60, 0.19, 0.65, 0.39)

plt.savefig("figures/fig1a_computational_graph_nature.svg", dpi=300,
            bbox_inches='tight')
print("✓ Saved Nature-style Fig 1a")
