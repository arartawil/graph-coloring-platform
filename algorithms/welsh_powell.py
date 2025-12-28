# Welsh-Powell Algorithm
import networkx as nx

def welsh_powell_coloring(graph):
    """
    Implements Welsh-Powell graph coloring algorithm.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        dict: Mapping of nodes to colors
    """
    # Sort nodes by degree in descending order
    sorted_nodes = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    
    coloring = {}
    color = 0
    
    while len(coloring) < len(graph.nodes()):
        # Find all nodes that can be colored with current color
        for node in sorted_nodes:
            if node in coloring:
                continue
            
            # Check if node can be colored with current color
            neighbor_colors = {coloring.get(neighbor) for neighbor in graph.neighbors(node)}
            
            if color not in neighbor_colors:
                coloring[node] = color
        
        color += 1
    
    return coloring
