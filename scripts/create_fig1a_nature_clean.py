import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25",
        linewidth=1,
        edgecolor="#444444",
        facecolor="#F5F5F5"
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9
    )

def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1, color="#222222")
    )

# Main pipeline (top row)
y_top = 3.5
w = 1.6
h = 1.0

x_design   = 0.5
x_mat      = 2.2
x_fdtd     = 3.9
x_field    = 5.6
x_signal   = 7.3
x_loss     = 9.0

box(x_design, y_top, w, h, "Design\nε(x,y)")
box(x_mat,    y_top, w, h, "Material\nmodel")
box(x_fdtd,   y_top, w, h, "FDTD\nsolver")
box(x_field,  y_top, w, h, "Field\nE(x,y,t)")
box(x_signal, y_top, w, h, "Signal\n∫|E(r)|²dt")
box(x_loss,   y_top, w, h, "Loss\nℒ")

# Gradient & update (stacked on right)
y_grad = 1.9
box(x_loss,   y_grad, w, h, "Gradient\n∂ℒ/∂ε")
box(x_loss,   0.3,   w, h, "Optimizer\nAdam")

# Arrows along main chain
arrow(x_design + w, y_top + h/2, x_mat,    y_top + h/2)
arrow(x_mat    + w, y_top + h/2, x_fdtd,   y_top + h/2)
arrow(x_fdtd   + w, y_top + h/2, x_field,  y_top + h/2)
arrow(x_field  + w, y_top + h/2, x_signal, y_top + h/2)
arrow(x_signal + w, y_top + h/2, x_loss,   y_top + h/2)

# Gradient path
arrow(x_loss + w/2, y_top,        x_loss + w/2, y_grad + h)
arrow(x_loss + w/2, y_grad,       x_loss + w/2, 0.3 + h)
# Update back to design
arrow(x_loss,       0.3 + h/2,    x_design + w/2, y_top)

# MD branch (bottom row)
y_md = 1.2
box(x_fdtd, y_md, w, h, "MD\nsimulation")
box(x_field, y_md, w, h, "Trajectory\nr(t)")

# MD arrows
arrow(x_fdtd + w/2, y_top,     x_fdtd + w/2, y_md + h)
arrow(x_fdtd + w,   y_md + h/2, x_field,     y_md + h/2)
arrow(x_field + w/2, y_md + h, x_field + w/2, y_top)

# Panel label
ax.text(0.1, 5.5, "(a)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig1a_computational_graph_clean.svg", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig1a_computational_graph_clean.png", dpi=600, bbox_inches="tight")
print("Saved: figures/fig1a_computational_graph_clean.[svg,png]")
