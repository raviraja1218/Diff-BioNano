from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

with Diagram("Computational Graph", show=False, direction="LR"):
    
    with Cluster("Design"):
        design = Custom("ε(x,y)", "./blank.png")
    
    material = Custom("Material\nmodel", "./blank.png")
    fdtd = Custom("FDTD\nsolver", "./blank.png")
    field = Custom("Field\nE(x,y,t)", "./blank.png")
    signal = Custom("Signal\n∫|E(r)|²dt", "./blank.png")
    loss = Custom("Loss ℒ", "./blank.png")

    with Cluster("Molecular\nDynamics"):
        md = Custom("MD\nsimulation", "./blank.png")
        traj = Custom("Trajectory\nr(t)", "./blank.png")

    grad = Custom("Gradient\n∂ℒ/∂ε", "./blank.png")
    opt = Custom("Optimizer\nAdam", "./blank.png")

    # Top path
    design >> material >> fdtd >> field >> signal >> loss

    # MD path
    md >> traj
    traj >> field
    fdtd >> md

    # Backprop
    loss >> grad >> opt >> design
