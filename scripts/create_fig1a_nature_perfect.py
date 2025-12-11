import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, text):
    r = patches.FancyBboxPatch(
        (x, y), 2.0, 1.1,
        boxstyle="round,pad=0.20",
        linewidth=1.4,
        edgecolor="black",
        facecolor="#F5F5F5"
    )
    ax.add_patch(r)
    ax.text(x+1.0, y+0.55, text, ha="center", va="center", fontsize=9)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.2))

yt = 4.2
yb = 2.2

# TOP ROW (clean)
box(0.5, yt, "Design\nε(x,y)")
box(3.0, yt, "Material\nmodel")
box(5.5, yt, "FDTD\nsolver")
box(8.0, yt, "Field\nE(x,y,t)")
box(10.5, yt, "Signal\n∫|E(r)|²dt")
box(13.0, yt, "Loss\nℒ")

arrow(2.5, yt+0.55, 3.0, yt+0.55)
arrow(5.0, yt+0.55, 5.5, yt+0.55)
arrow(7.5, yt+0.55, 8.0, yt+0.55)
arrow(10.0, yt+0.55, 10.5, yt+0.55)
arrow(12.5, yt+0.55, 13.0, yt+0.55)

# MD Block (Bottom)
box(5.5, yb, "MD\nsimulation")
box(8.0, yb, "Trajectory\nr(t)")

arrow(6.5, yb+1.1, 6.5, yt)   # MD → FDTD
arrow(8.0+1.0, yb+1.1, 8.0+1.0, yt)  # Trajectory → Field
arrow(7.5, yb+0.55, 8.0, yb+0.55)     # MD → Trajectory

# Gradient + Optimizer (stacked right side)
box(13.0, 3.0, "Gradient\n∂ℒ/∂ε")
box(13.0, 1.5, "Optimizer\nAdam")

arrow(13.0+1.0, 3.0+0.55, 13.0+1.0, yt)
arrow(13.0+1.0, 1.5+0.55, 13.0+1.0, 3.0)

# Feedback (CLEAN horizontal)
arrow(13.0, 1.5+0.55, 0.5+2.0, yt+0.55)

# Panel label
ax.text(0.1, 5.3, "(a)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig1a_computational_graph_perfect.png", dpi=600)
plt.savefig("figures/fig1a_computational_graph_perfect.svg", dpi=300)
print("DONE: Nature-quality figure saved.")
