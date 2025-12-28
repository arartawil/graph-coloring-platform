"""Largest First graph coloring algorithm."""

from __future__ import annotations

import time
from typing import Any, Dict

import networkx as nx


def largest_first_coloring(graph: nx.Graph) -> Dict[Any, int]:
    """
    Largest First (LF) algorithm for graph coloring.
    
    Colors vertices in order of decreasing degree (largest first).
    Simple heuristic that often performs well.
    
    Time Complexity: O(n log n + m)
    Space Complexity: O(n)
    
    Args:
        graph: NetworkX graph
    
    Returns:
        Mapping of node -> color (ints starting at 0)
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    # Sort nodes by degree descending
    ordering = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    
    coloring: Dict[Any, int] = {}
    
    for node in ordering:
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


class LargestFirstAlgorithm:
    """Largest First algorithm with metrics tracking."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.coloring: Dict[Any, int] = {}
        self.execution_time: float = 0.0
    
    def color(self) -> Dict[Any, int]:
        """Execute the Largest First algorithm."""
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring
        
        start = time.perf_counter()
        
        # Sort nodes by degree descending
        ordering = sorted(
            self.graph.nodes(),
            key=lambda n: self.graph.degree(n),
            reverse=True
        )
        
        coloring: Dict[Any, int] = {}
        
        for node in ordering:
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
    "largest_first_coloring",
    "LargestFirstAlgorithm",
]
