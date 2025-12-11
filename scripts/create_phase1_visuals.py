import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from graphviz import Digraph

# ============================================================
# FIGURE 1A — Computational Graph Diagram (Graphviz)
# ============================================================

print("Generating Figure 1a: Computational Graph...")

dot = Digraph('DiffBioNano', format='svg')
dot.attr(rankdir='TB')

# Nodes
dot.node('Design', 'Design Parameters\nε(x,y)', shape='box', style='filled', fillcolor='lightblue')
dot.node('Material', 'Material Grid', shape='box', style='filled', fillcolor='lightgreen')
dot.node('FDTD', 'FDTD Solver', shape='box', style='filled', fillcolor='orange')
dot.node('Efield', 'E(x,y,t)', shape='ellipse', style='filled', fillcolor='yellow')
dot.node('MD', 'Molecular Dynamics', shape='box', style='filled', fillcolor='pink')
dot.node('Trajectory', 'rₘ(t)', shape='ellipse', style='filled', fillcolor='lightpink')
dot.node('Interp', 'Field Interpolation\nE(rₘ(t))', shape='diamond', style='filled', fillcolor='cyan')
dot.node('Signal', 'Signal\n∫|E(rₘ)|²dt', shape='ellipse', style='filled', fillcolor='lightyellow')
dot.node('Loss', 'Loss ℒ', shape='box', style='filled', fillcolor='red')
dot.node('Grad', 'Gradient\n∂ℒ/∂ε', shape='parallelogram', style='filled', fillcolor='purple')
dot.node('Update', 'Adam Update', shape='box', style='filled', fillcolor='green')

# Edges
dot.edge('Design', 'Material')
dot.edge('Material', 'FDTD')
dot.edge('FDTD', 'Efield')
dot.edge('MD', 'Trajectory')
dot.edge('Efield', 'Interp')
dot.edge('Trajectory', 'Interp')
dot.edge('Interp', 'Signal')
dot.edge('Signal', 'Loss')
dot.edge('Loss', 'Grad')
dot.edge('Grad', 'Update')
dot.edge('Update', 'Design')

dot.render('figures/fig1a_computational_graph', cleanup=True)

print("✓ Saved: figures/fig1a_computational_graph.svg")


# ============================================================
# FIGURE 1B — Alanine Dipeptide 3D
# ============================================================

print("Generating Figure 1b: Alanine Dipeptide 3D...")

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Simplified coordinates
x = [0, 1.5, 2.5, 3.5, 1.5]
y = [0, 0, 1.4, 1.4, -0.5]
z = [0, 0, 0, 0, 1.2]
colors = ['blue', 'gray', 'gray', 'red', 'gray']
labels = ['N', 'CA', 'C', 'O', 'CB']

# Plot atoms
for i in range(5):
    ax.scatter(x[i], y[i], z[i], s=120, c=colors[i], edgecolor='black')
    ax.text(x[i], y[i], z[i], labels[i], fontsize=8)

# Bonds
bonds = [(0,1), (1,2), (2,3), (1,4)]
for i, j in bonds:
    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k-', linewidth=2)

ax.set_title("Alanine Dipeptide — 3D Structure")
ax.set_axis_off()

plt.savefig("figures/fig1b_molecule_3d.png", dpi=600, bbox_inches='tight')
print("✓ Saved: figures/fig1b_molecule_3d.png")


# ============================================================
# FIGURE S1 — Derivation Flowchart
# ============================================================

print("Generating Supplementary Figure S1: Derivation Flow...")

fig, ax = plt.subplots(figsize=(8, 5))

def draw_box(ax, x, y, w, h, text, color):
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

# Boxes
draw_box(ax, 0.35, 0.75, 0.3, 0.15, "Lagrangian\nℒ", "lightblue")
draw_box(ax, 0.35, 0.5, 0.3, 0.15, "Hamiltonian\nℋ", "lightgreen")
draw_box(ax, 0.1, 0.25, 0.25, 0.15, "ℋ_MD", "orange")
draw_box(ax, 0.375, 0.25, 0.25, 0.15, "ℋ_int", "yellow")
draw_box(ax, 0.65, 0.25, 0.25, 0.15, "ℋ_EM", "pink")

# Arrows
ax.annotate("", xy=(0.5, 0.65), xytext=(0.5, 0.75), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.225, 0.4), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.5, 0.4), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.775, 0.4), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", lw=2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig("supplementary/figS1_derivation.png", dpi=600, bbox_inches='tight')
print("✓ Saved: supplementary/figS1_derivation.png")

print("\nALL PHASE 1 VISUALS GENERATED SUCCESSFULLY ✔")
