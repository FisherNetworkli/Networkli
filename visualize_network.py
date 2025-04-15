import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

# Create a graph with 5 nodes
G = nx.Graph()

# Add nodes
nodes = list(range(5))
G.add_nodes_from(nodes)

# Add random edges with 60% probability
for i in range(5):
    for j in range(i+1, 5):
        if np.random.random() < 0.6:  # 60% chance of connection
            G.add_edge(i, j)

# Create the visualization
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

# Draw the network
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=500, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

# Add a title
plt.title("Network Visualization of 5 People", pad=20, fontsize=16)

# Remove axis
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()

# Print network statistics
print("\nNetwork Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print("\nNode degrees:")
for node in G.nodes():
    print(f"Person {node}: {G.degree(node)} connections") 