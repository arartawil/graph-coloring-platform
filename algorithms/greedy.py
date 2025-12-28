# Greedy Graph Coloring Algorithm
import networkx as nx

def greedy_coloring(graph):
    """
    Implements greedy graph coloring algorithm.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        dict: Mapping of nodes to colors
    """
    coloring = {}
    
    for node in graph.nodes():
        # Get colors of neighbors
        neighbor_colors = {coloring[neighbor] for neighbor in graph.neighbors(node) if neighbor in coloring}
        
        # Find the minimum available color
        color = 0
        while color in neighbor_colors:
            color += 1
        
        coloring[node] = color
    
    return coloring
