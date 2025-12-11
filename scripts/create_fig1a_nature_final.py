import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6.2, 3.4))  # Nature panel size
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, text):
    w, h = 2.0, 1.0
    r = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18",
        linewidth=1.2,
        edgecolor="black",
        facecolor="#F2F2F2"
    )
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="black"))

# === MAIN PIPELINE (TOP ROW) ===
yt = 4.0

box(0.3,  yt, "Design\nε(x,y)")
box(2.7,  yt, "Material\nmodel")
box(5.1,  yt, "FDTD\nsolver")
box(7.5,  yt, "Field\nE(x,y,t)")
box(9.9,  yt, "Signal\n∫|E(r)|²dt")
box(12.3, yt, "Loss\nℒ")

arrow(2.7,  yt+0.5, 0.3+2.0, yt+0.5)
arrow(5.1,  yt+0.5, 2.7+2.0, yt+0.5)
arrow(7.5,  yt+0.5, 5.1+2.0, yt+0.5)
arrow(9.9,  yt+0.5, 7.5+2.0, yt+0.5)
arrow(12.3, yt+0.5, 9.9+2.0, yt+0.5)

# === GRADIENT BLOCK ===
box(12.3, 2.2, "Gradient\n∂ℒ/∂ε")
arrow(13.3, 3.2, 13.3, 4.0)

box(12.3, 0.6, "Optimizer\nAdam")
arrow(13.3, 1.6, 13.3, 2.2)

# Return to design
arrow(12.3, 1.0, 1.3, 4.0)

# === MD BRANCH (BOTTOM ROW) ===
yb = 2.1

box(5.1, yb, "MD\nsimulation")
box(7.5, yb, "Trajectory\nr(t)")

arrow(6.1, yb+1.0, 6.1, yt)        # MD → FDTD
arrow(7.5, yb+0.5, 5.1+2.0, yb+0.5)
arrow(8.5, yb+1.0, 8.5, yt)        # Trajectory → Field

# Panel label
ax.text(0.0, 5.5, "(a)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig1a_computational_graph_final.svg", dpi=350)
plt.savefig("figures/fig1a_computational_graph_final.png", dpi=600)
print("DONE → Nature figure saved.")
