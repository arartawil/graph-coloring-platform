# DSatur (Degree of Saturation) Algorithm
import networkx as nx

def dsatur_coloring(graph):
    """
    Implements DSatur graph coloring algorithm.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        dict: Mapping of nodes to colors
    """
    coloring = {}
    uncolored = set(graph.nodes())
    
    # Start with the node of highest degree
    first_node = max(uncolored, key=lambda n: graph.degree(n))
    coloring[first_node] = 0
    uncolored.remove(first_node)
    
    while uncolored:
        # Calculate saturation degree for each uncolored node
        max_saturation = -1
        selected_node = None
        
        for node in uncolored:
            # Count unique colors in neighbors
            neighbor_colors = {coloring[neighbor] for neighbor in graph.neighbors(node) if neighbor in coloring}
            saturation = len(neighbor_colors)
            
            if saturation > max_saturation or (saturation == max_saturation and selected_node is None):
                max_saturation = saturation
                selected_node = node
            elif saturation == max_saturation:
                # Tie-breaking: choose node with higher degree
                if graph.degree(node) > graph.degree(selected_node):
                    selected_node = node
        
        # Color the selected node
        neighbor_colors = {coloring[neighbor] for neighbor in graph.neighbors(selected_node) if neighbor in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        
        coloring[selected_node] = color
        uncolored.remove(selected_node)
    
    return coloring
