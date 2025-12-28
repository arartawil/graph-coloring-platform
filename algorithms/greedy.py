"""Greedy graph coloring algorithms and utilities."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def greedy_coloring(graph: nx.Graph, node_order: Optional[Iterable[Any]] = None) -> Dict[Any, int]:
    """Color the graph greedily using the lowest available color.

    Args:
        graph: NetworkX graph (directed treated as undirected for coloring).
        node_order: Optional iterable defining the visitation order. Defaults to graph nodes order.

    Returns:
        Mapping of node -> color (ints starting at 0).
    """
    if graph.number_of_nodes() == 0:
        return {}

    ordering = list(node_order) if node_order is not None else list(graph.nodes())
    coloring: Dict[Any, int] = {}

    for node in ordering:
        neighbor_colors = set()
        # Local variable for speed on large graphs
        for nbr in graph.neighbors(node):
            if nbr in coloring:
                neighbor_colors.add(coloring[nbr])

        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[node] = color

    return coloring


def greedy_with_order(graph: nx.Graph, order_strategy: str = "default") -> Dict[Any, int]:
    """Run greedy coloring with a chosen ordering strategy.

    Strategies:
        - default: as provided by the graph iterator
        - random: random permutation
        - degree_desc: sort by degree descending
        - degree_asc: sort by degree ascending
    """
    nodes = list(graph.nodes())
    if order_strategy == "random":
        random.shuffle(nodes)
    elif order_strategy == "degree_desc":
        nodes.sort(key=lambda n: graph.degree(n), reverse=True)
    elif order_strategy == "degree_asc":
        nodes.sort(key=lambda n: graph.degree(n))

    return greedy_coloring(graph, node_order=nodes)


def calculate_metrics(graph: nx.Graph, coloring: Dict[Any, int]) -> Dict[str, Any]:
    """Calculate metrics for a produced coloring.

    Returns keys: colors_used, is_valid, execution_time (None if unknown),
    chromatic_number (if present in graph.graph), optimal (if chromatic known).
    """
    start = time.perf_counter()
    # Validation
    is_valid = True
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            is_valid = False
            break
    colors_used = len(set(coloring.values())) if coloring else 0
    exec_time = time.perf_counter() - start

    chromatic = graph.graph.get("chromatic_number")
    optimal = None
    if chromatic is not None:
        optimal = colors_used == chromatic

    return {
        "colors_used": colors_used,
        "is_valid": is_valid,
        "execution_time": exec_time,
        "chromatic_number": chromatic,
        "optimal": optimal,
    }


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------


class GreedyColoringAlgorithm:
    """Greedy coloring with optional step tracking for visualization."""

    def __init__(self, graph: nx.Graph, order_strategy: str = "default", track_steps: bool = False):
        self.graph = graph
        self.order_strategy = order_strategy
        self.track_steps = track_steps
        self.coloring: Dict[Any, int] = {}
        self.steps: List[Dict[str, Any]] = []
        self.execution_time: float = 0.0

    def color(self) -> Dict[Any, int]:
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring

        ordering = list(self.graph.nodes())
        if self.order_strategy == "random":
            random.shuffle(ordering)
        elif self.order_strategy == "degree_desc":
            ordering.sort(key=lambda n: self.graph.degree(n), reverse=True)
        elif self.order_strategy == "degree_asc":
            ordering.sort(key=lambda n: self.graph.degree(n))

        start = time.perf_counter()
        coloring: Dict[Any, int] = {}
        steps: List[Dict[str, Any]] = [] if self.track_steps and self.graph.number_of_nodes() <= 1000 else None

        for node in ordering:
            neighbor_colors = set()
            for nbr in self.graph.neighbors(node):
                if nbr in coloring:
                    neighbor_colors.add(coloring[nbr])

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[node] = color

            if steps is not None:
                steps.append({
                    "node": node,
                    "color": color,
                    "excluded": sorted(neighbor_colors),
                })

        self.execution_time = time.perf_counter() - start
        self.coloring = coloring
        self.steps = steps or []
        return self.coloring

    def get_metrics(self) -> Dict[str, Any]:
        if not self.coloring:
            self.color()
        metrics = calculate_metrics(self.graph, self.coloring)
        metrics["execution_time"] = self.execution_time or metrics["execution_time"]
        return metrics

    def visualize_steps(self) -> List[Dict[str, Any]]:
        """Return recorded steps for animation consumers."""
        return self.steps
