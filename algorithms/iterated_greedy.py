"""Iterated Greedy coloring algorithm."""

from __future__ import annotations

import random
import time
from typing import Any, Dict

import networkx as nx


def iterated_greedy_coloring(graph: nx.Graph, iterations: int = 5, seed: int = None) -> Dict[Any, int]:
    """
    Iterated Greedy (IG) algorithm for graph coloring.
    
    Runs greedy coloring multiple times with different random orderings
    and returns the best result.
    
    Time Complexity: O(k * (n + m)) where k is iterations
    Space Complexity: O(n)
    
    Args:
        graph: NetworkX graph
        iterations: Number of greedy iterations to try
        seed: Random seed for reproducibility
    
    Returns:
        Mapping of node -> color (ints starting at 0)
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    if seed is not None:
        random.seed(seed)
    
    best_coloring = None
    best_colors = float('inf')
    
    for _ in range(iterations):
        # Random permutation of nodes
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        
        # Greedy coloring with this ordering
        coloring: Dict[Any, int] = {}
        
        for node in nodes:
            neighbor_colors = set()
            for neighbor in graph.neighbors(node):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[node] = color
        
        # Track best result
        num_colors = len(set(coloring.values()))
        if num_colors < best_colors:
            best_colors = num_colors
            best_coloring = coloring
    
    return best_coloring if best_coloring is not None else {}


class IteratedGreedyAlgorithm:
    """Iterated Greedy algorithm with metrics tracking."""
    
    def __init__(self, graph: nx.Graph, iterations: int = 5):
        self.graph = graph
        self.iterations = iterations
        self.coloring: Dict[Any, int] = {}
        self.execution_time: float = 0.0
    
    def color(self) -> Dict[Any, int]:
        """Execute the Iterated Greedy algorithm."""
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring
        
        start = time.perf_counter()
        
        best_coloring = None
        best_colors = float('inf')
        
        for _ in range(self.iterations):
            # Random permutation of nodes
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)
            
            # Greedy coloring
            coloring: Dict[Any, int] = {}
            
            for node in nodes:
                neighbor_colors = set()
                for neighbor in self.graph.neighbors(node):
                    if neighbor in coloring:
                        neighbor_colors.add(coloring[neighbor])
                
                color = 0
                while color in neighbor_colors:
                    color += 1
                
                coloring[node] = color
            
            # Track best
            num_colors = len(set(coloring.values()))
            if num_colors < best_colors:
                best_colors = num_colors
                best_coloring = coloring
        
        self.execution_time = time.perf_counter() - start
        self.coloring = best_coloring if best_coloring is not None else {}
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
    "iterated_greedy_coloring",
    "IteratedGreedyAlgorithm",
]
