# Tabu Search for Graph Coloring
import networkx as nx
import random

def tabu_search_coloring(graph, max_iterations=1000, tabu_tenure=10, num_colors=None):
    """
    Implements Tabu Search for graph coloring.
    
    Args:
        graph: NetworkX graph object
        max_iterations: Maximum number of iterations
        tabu_tenure: How long a move stays in tabu list
        num_colors: Target number of colors (if None, uses initial greedy coloring)
        
    Returns:
        dict: Mapping of nodes to colors
    """
    from .greedy import greedy_coloring
    
    # Initialize with greedy coloring
    current_solution = greedy_coloring(graph)
    
    if num_colors is None:
        num_colors = max(current_solution.values()) + 1
    
    best_solution = current_solution.copy()
    best_conflicts = count_conflicts(graph, best_solution)
    
    tabu_list = []
    
    for iteration in range(max_iterations):
        # Find best neighbor not in tabu list
        best_neighbor = None
        best_neighbor_conflicts = float('inf')
        
        for node in graph.nodes():
            old_color = current_solution[node]
            
            for new_color in range(num_colors):
                if new_color == old_color:
                    continue
                
                move = (node, old_color, new_color)
                
                # Check if move is in tabu list
                if move in tabu_list:
                    continue
                
                # Apply move temporarily
                current_solution[node] = new_color
                conflicts = count_conflicts(graph, current_solution)
                
                if conflicts < best_neighbor_conflicts:
                    best_neighbor = move
                    best_neighbor_conflicts = conflicts
                
                # Revert move
                current_solution[node] = old_color
        
        if best_neighbor is None:
            break
        
        # Apply best move
        node, old_color, new_color = best_neighbor
        current_solution[node] = new_color
        
        # Update tabu list
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update best solution
        if best_neighbor_conflicts < best_conflicts:
            best_solution = current_solution.copy()
            best_conflicts = best_neighbor_conflicts
        
        if best_conflicts == 0:
            break
    
    return best_solution

def count_conflicts(graph, coloring):
    """Count the number of edge conflicts in a coloring."""
    conflicts = 0
    for u, v in graph.edges():
        if coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts
