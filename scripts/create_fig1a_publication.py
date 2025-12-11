from graphviz import Digraph

dot = Digraph('Fig1a', format='svg')
dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.8')

style = dict(shape='box', style='rounded,filled', fillcolor='white', color='black')

node = lambda name,label: dot.node(name,label,**style)

node('Design', 'Design Params\nε(x,y)')
node('Material', 'Material Grid')
node('FDTD', 'FDTD Solver')
node('Field', 'E(x,y,t)')
node('MD', 'Molecular Dynamics')
node('Traj', 'rₘ(t)')
node('Interp', 'Field Interpolation')
node('Signal', 'Signal\n∫|E(rₘ)|²dt')
node('Loss', 'Loss ℒ')
node('Grad', 'Gradient')
node('Update', 'Adam Update')

dot.edge('Design','Material')
dot.edge('Material','FDTD')
dot.edge('FDTD','Field')
dot.edge('MD','Traj')
dot.edge('Field','Interp')
dot.edge('Traj','Interp')
dot.edge('Interp','Signal')
dot.edge('Signal','Loss')
dot.edge('Loss','Grad')
dot.edge('Grad','Update')
dot.edge('Update','Design')

dot.render('figures/fig1a_computational_graph_publication', cleanup=True)
print("✓ Saved: figures/fig1a_computational_graph_publication.svg")
