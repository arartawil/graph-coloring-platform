"""DSatur (Degree of Saturation) graph coloring implementation."""

from __future__ import annotations

import heapq
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


def calculate_saturation(graph: nx.Graph, coloring: Dict[Any, int], uncolored_nodes: Iterable[Any]) -> Dict[Any, int]:
    """Calculate saturation (distinct neighbor colors) for each uncolored node."""
    saturation: Dict[Any, int] = {}
    for node in uncolored_nodes:
        neighbor_colors = {coloring[n] for n in graph.neighbors(node) if n in coloring}
        saturation[node] = len(neighbor_colors)
    return saturation


def dsatur_coloring(graph: nx.Graph) -> Dict[Any, int]:
    """Functional wrapper for DSatur coloring."""
    algo = DSaturAlgorithm(graph)
    return algo.color()


class DSaturAlgorithm:
    """DSatur algorithm with heap-based selection and step logging."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.coloring: Dict[Any, int] = {}
        self.steps: List[Dict[str, Any]] = []
        self.execution_time: float = 0.0

    def color(self) -> Dict[Any, int]:
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring

        start = time.perf_counter()
        uncolored: Set[Any] = set(self.graph.nodes())
        neighbor_color_sets: Dict[Any, Set[int]] = {n: set() for n in self.graph.nodes()}

        # seed with highest degree vertex
        first_node = max(uncolored, key=lambda n: self.graph.degree(n))
        self.coloring[first_node] = 0
        uncolored.remove(first_node)
        self._update_neighbor_colors(first_node, 0, neighbor_color_sets, uncolored)
        self.steps.append({"node": first_node, "color": 0, "saturation": 0})

        heap = self._build_heap(uncolored, neighbor_color_sets)

        while uncolored:
            _, _, _, node = heapq.heappop(heap)
            if node not in uncolored:
                continue  # stale entry

            neighbor_colors = neighbor_color_sets[node]
            color = 0
            while color in neighbor_colors:
                color += 1

            self.coloring[node] = color
            uncolored.remove(node)
            self._update_neighbor_colors(node, color, neighbor_color_sets, uncolored)
            self.steps.append({
                "node": node,
                "color": color,
                "saturation": len(neighbor_colors),
            })

            # push updated neighbors to heap
            for nbr in self.graph.neighbors(node):
                if nbr in uncolored:
                    sat = len(neighbor_color_sets[nbr])
                    deg = self.graph.degree(nbr)
                    heapq.heappush(heap, (-sat, -deg, self._tie_break(nbr), nbr))

        self.execution_time = time.perf_counter() - start
        return self.coloring

    def get_saturation_degrees(self) -> Dict[Any, int]:
        if not self.coloring:
            return calculate_saturation(self.graph, {}, self.graph.nodes())
        return calculate_saturation(self.graph, self.coloring, [n for n in self.graph.nodes() if n not in self.coloring])

    def select_next_vertex(self, uncolored: Set[Any], neighbor_color_sets: Dict[Any, Set[int]]) -> Any:
        """Select next vertex using saturation then degree tie-break."""
        heap = self._build_heap(uncolored, neighbor_color_sets)
        while heap:
            _, _, _, node = heapq.heappop(heap)
            if node in uncolored:
                return node
        return None

    def visualize_process(self) -> List[Dict[str, Any]]:
        return self.steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_heap(self, uncolored: Set[Any], neighbor_color_sets: Dict[Any, Set[int]]):
        heap: List[Tuple[int, int, Any, Any]] = []
        for node in uncolored:
            sat = len(neighbor_color_sets[node])
            deg = self.graph.degree(node)
            heapq.heappush(heap, (-sat, -deg, self._tie_break(node), node))
        return heap

    def _update_neighbor_colors(
        self,
        node: Any,
        color: int,
        neighbor_color_sets: Dict[Any, Set[int]],
        uncolored: Set[Any],
    ) -> None:
        for nbr in self.graph.neighbors(node):
            if nbr in uncolored:
                neighbor_color_sets[nbr].add(color)

    def _tie_break(self, node: Any) -> Any:
        """Stable tie-breaker to keep heap deterministic."""
        return str(node)


__all__ = [
    "dsatur_coloring",
    "calculate_saturation",
    "DSaturAlgorithm",
]
