"""Smallest Last graph coloring algorithm."""

from __future__ import annotations

import time
from typing import Any, Dict

import networkx as nx


def smallest_last_coloring(graph: nx.Graph) -> Dict[Any, int]:
    """
    Smallest Last (SL) algorithm for graph coloring.
    
    Removes vertices in order of increasing degree, then colors them
    in reverse order. Often produces better results than greedy.
    
    Time Complexity: O(n + m)
    Space Complexity: O(n)
    
    Args:
        graph: NetworkX graph
    
    Returns:
        Mapping of node -> color (ints starting at 0)
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    # Make a copy to avoid modifying original
    G = graph.copy()
    ordering = []
    
    # Remove vertices in order of smallest degree
    while G.number_of_nodes() > 0:
        # Find vertex with minimum degree
        min_degree_node = min(G.nodes(), key=lambda n: G.degree(n))
        ordering.append(min_degree_node)
        G.remove_node(min_degree_node)
    
    # Color in reverse order (largest degree first in original graph)
    coloring: Dict[Any, int] = {}
    
    for node in reversed(ordering):
        # Find colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        
        # Assign smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        
        coloring[node] = color
    
    return coloring


class SmallestLastAlgorithm:
    """Smallest Last algorithm with metrics tracking."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.coloring: Dict[Any, int] = {}
        self.execution_time: float = 0.0
        self.ordering: list = []
    
    def color(self) -> Dict[Any, int]:
        """Execute the Smallest Last algorithm."""
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring
        
        start = time.perf_counter()
        
        # Copy graph
        G = self.graph.copy()
        self.ordering = []
        
        # Remove vertices in order of smallest degree
        while G.number_of_nodes() > 0:
            min_degree_node = min(G.nodes(), key=lambda n: G.degree(n))
            self.ordering.append(min_degree_node)
            G.remove_node(min_degree_node)
        
        # Color in reverse order
        coloring: Dict[Any, int] = {}
        
        for node in reversed(self.ordering):
            neighbor_colors = set()
            for neighbor in self.graph.neighbors(node):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[node] = color
        
        self.execution_time = time.perf_counter() - start
        self.coloring = coloring
        return self.coloring
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get algorithm metrics."""
        if not self.coloring:
            self.color()
        
        colors_used = len(set(self.coloring.values())) if self.coloring else 0
        valid = all(
            self.coloring.get(u) != self.coloring.get(v) 
            for u, v in self.graph.edges()
        )
        chromatic = self.graph.graph.get("chromatic_number")
        optimal = None if chromatic is None else colors_used == chromatic
        
        return {
            "colors_used": colors_used,
            "execution_time": self.execution_time,
            "is_valid": valid,
            "chromatic_number": chromatic,
            "optimal": optimal,
        }


__all__ = [
    "smallest_last_coloring",
    "SmallestLastAlgorithm",
]
