"""Welsh-Powell graph coloring implementation."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Set

import networkx as nx


def sort_by_degree(graph: nx.Graph) -> List[Any]:
    """Return vertices sorted by degree descending with deterministic ties."""
    return sorted(graph.nodes(), key=lambda n: (graph.degree(n), str(n)), reverse=True)


def find_independent_set(
    graph: nx.Graph,
    colored: Dict[Any, int],
    available_vertices: Iterable[Any],
    color: int,
) -> Set[Any]:
    """Greedily build a maximal independent set from available vertices."""
    chosen: Set[Any] = set()
    for node in available_vertices:
        if node in colored:
            continue
        if all(neigh not in chosen for neigh in graph.neighbors(node)):
            chosen.add(node)
            colored[node] = color
    return chosen


def welsh_powell_coloring(graph: nx.Graph) -> Dict[Any, int]:
    """Functional wrapper for Welsh-Powell coloring."""
    algo = WelshPowellAlgorithm(graph)
    return algo.color()


class WelshPowellAlgorithm:
    """Welsh-Powell coloring with metrics tracking."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.coloring: Dict[Any, int] = {}
        self.execution_time: float = 0.0

    def sort_vertices(self) -> List[Any]:
        return sort_by_degree(self.graph)

    def color_independent_set(self, vertices: Iterable[Any], color: int) -> Set[Any]:
        return find_independent_set(self.graph, self.coloring, vertices, color)

    def color(self) -> Dict[Any, int]:
        if self.graph.number_of_nodes() == 0:
            self.coloring = {}
            self.execution_time = 0.0
            return self.coloring

        ordered = self.sort_vertices()
        color = 0
        start = time.perf_counter()

        while len(self.coloring) < self.graph.number_of_nodes():
            self.color_independent_set(ordered, color)
            color += 1

        self.execution_time = time.perf_counter() - start
        return self.coloring

    def get_metrics(self) -> Dict[str, Any]:
        if not self.coloring:
            self.color()
        colors_used = len(set(self.coloring.values())) if self.coloring else 0
        valid = all(self.coloring.get(u) != self.coloring.get(v) for u, v in self.graph.edges())
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
    "welsh_powell_coloring",
    "sort_by_degree",
    "find_independent_set",
    "WelshPowellAlgorithm",
]
