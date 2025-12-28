# Graph Operations Utilities
import networkx as nx
import numpy as np

def validate_coloring(graph, coloring):
    """
    Validate that a coloring is proper (no adjacent nodes have same color).
    
    Args:
        graph: NetworkX graph
        coloring: Dictionary mapping nodes to colors
        
    Returns:
        bool: True if coloring is valid
    """
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True

def count_colors(coloring):
    """
    Count the number of unique colors used.
    
    Args:
        coloring: Dictionary mapping nodes to colors
        
    Returns:
        int: Number of unique colors
    """
    return len(set(coloring.values()))

def chromatic_number_upper_bound(graph):
    """
    Calculate upper bound for chromatic number (max degree + 1).
    
    Args:
        graph: NetworkX graph
        
    Returns:
        int: Upper bound
    """
    if len(graph.nodes()) == 0:
        return 0
    return max(dict(graph.degree()).values()) + 1

def graph_density(graph):
    """
    Calculate graph density.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        float: Graph density
    """
    return nx.density(graph)

def get_graph_stats(graph):
    """
    Get comprehensive statistics about a graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        dict: Dictionary of statistics
    """
    is_directed = graph.is_directed()
    base_graph = graph.to_undirected() if is_directed else graph

    node_count = base_graph.number_of_nodes()
    density = nx.density(base_graph) if node_count > 0 else 0.0
    connected = False
    if node_count > 0:
        try:
            connected = nx.is_connected(base_graph)
        except Exception:
            connected = False

    stats = {
        'num_nodes': node_count,
        'num_edges': base_graph.number_of_edges(),
        'density': density,
        'is_connected': connected,
        'directed': is_directed,
    }

    if node_count > 0:
        degrees = dict(base_graph.degree())
        stats['avg_degree'] = sum(degrees.values()) / node_count
        stats['max_degree'] = max(degrees.values())
        stats['min_degree'] = min(degrees.values())

    return stats

def compare_colorings(graph, coloring1, coloring2):
    """
    Compare two colorings of the same graph.
    
    Args:
        graph: NetworkX graph
        coloring1: First coloring
        coloring2: Second coloring
        
    Returns:
        dict: Comparison results
    """
    return {
        'colors_used_1': count_colors(coloring1),
        'colors_used_2': count_colors(coloring2),
        'valid_1': validate_coloring(graph, coloring1),
        'valid_2': validate_coloring(graph, coloring2),
        'difference': abs(count_colors(coloring1) - count_colors(coloring2))
    }

def convert_coloring_format(coloring):
    """
    Convert coloring from node->color to color->nodes format.
    
    Args:
        coloring: Dictionary mapping nodes to colors
        
    Returns:
        dict: Dictionary mapping colors to list of nodes
    """
    color_groups = {}
    for node, color in coloring.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(node)
    return color_groups
