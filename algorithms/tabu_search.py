"""Adaptive Tabu Search for graph coloring."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from .greedy import greedy_coloring


Move = Tuple[Any, int, int]


class TabuSearchColoring:
    """Adaptive Tabu Search with diversification and history tracking."""

    def __init__(
        self,
        graph: nx.Graph,
        max_iterations: int = 300,
        tabu_tenure: int = 30,
        aspiration_criterion: bool = True,
        no_improve_limit: int = 80,
    ):
        self.graph = graph
        self.max_iterations = max_iterations
        self.base_tabu_tenure = tabu_tenure
        self.aspiration_criterion = aspiration_criterion
        self.no_improve_limit = no_improve_limit

        self.coloring: Dict[Any, int] = {}
        self.best_coloring: Dict[Any, int] = {}
        self.best_conflicts: int = math.inf
        self.history: List[Dict[str, Any]] = []
        self.tabu_list: Deque[Tuple[Move, int]] = deque()
        self.long_term_memory: Dict[Any, int] = defaultdict(int)
        self.iteration = 0
        self.execution_time: float = 0.0

    def initial_solution(self) -> Dict[Any, int]:
        if self.graph.number_of_nodes() == 0:
            return {}
        return greedy_coloring(self.graph)

    def calculate_conflicts(self, coloring: Dict[Any, int]) -> int:
        conflicts = 0
        for u, v in self.graph.edges():
            if coloring.get(u) == coloring.get(v):
                conflicts += 1
        return conflicts

    def generate_neighbors(self, coloring: Dict[Any, int], num_colors: int) -> List[Tuple[Move, Dict[Any, int], int]]:
        neighbors = []
        conflict_edges = [(u, v) for u, v in self.graph.edges() if coloring.get(u) == coloring.get(v)]
        conflict_nodes = {u for edge in conflict_edges for u in edge} or set(self.graph.nodes())

        for node in conflict_nodes:
            current_color = coloring[node]
            for new_color in range(num_colors + 1):
                if new_color == current_color:
                    continue
                move: Move = (node, current_color, new_color)
                new_coloring = coloring.copy()
                new_coloring[node] = new_color
                conflicts = self.calculate_conflicts(new_coloring)
                neighbors.append((move, new_coloring, conflicts))
        return neighbors

    def is_tabu(self, move: Move, current_best: int, candidate_conflicts: int) -> bool:
        for m, expiry in list(self.tabu_list):
            if move == m and self.iteration <= expiry:
                if self.aspiration_criterion and candidate_conflicts < current_best:
                    return False
                return True
        return False

    def update_tabu_list(self, move: Move, tenure: int) -> None:
        expiry = self.iteration + tenure
        self.tabu_list.append((move, expiry))
        while self.tabu_list and self.tabu_list[0][1] < self.iteration:
            self.tabu_list.popleft()

    def adaptive_tenure(self, stagnation_counter: int) -> int:
        return max(5, int(self.base_tabu_tenure * (1 + stagnation_counter / max(1, self.no_improve_limit))))

    def diversify(self, coloring: Dict[Any, int]) -> Dict[Any, int]:
        diversified = coloring.copy()
        # Recolor a small set of high-frequency nodes to escape local minima
        if not self.long_term_memory:
            return diversified
        sorted_nodes = sorted(self.long_term_memory.items(), key=lambda kv: kv[1], reverse=True)
        top_nodes = [n for n, _ in sorted_nodes[: max(1, len(sorted_nodes) // 10)]]
        for node in top_nodes:
            diversified[node] = random.randint(0, max(diversified.values()) + 1)
        return diversified

    def intensify(self, coloring: Dict[Any, int]) -> Dict[Any, int]:
        # Greedy re-color on current ordering to refine solution
        order = sorted(self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True)
        return greedy_coloring(self.graph, node_order=order)

    def color(self) -> Dict[Any, int]:
        if self.graph.number_of_nodes() == 0:
            self.best_coloring = {}
            self.best_conflicts = 0
            return {}

        start = time.perf_counter()
        current = self.initial_solution()
        num_colors = max(current.values()) + 1 if current else 1
        current_conflicts = self.calculate_conflicts(current)

        self.best_coloring = current.copy()
        self.best_conflicts = current_conflicts
        stagnation = 0

        for self.iteration in range(1, self.max_iterations + 1):
            neighbors = self.generate_neighbors(current, num_colors)
            neighbors.sort(key=lambda x: x[2])  # ascending by conflicts

            chosen_move = None
            chosen_coloring = None
            chosen_conflicts = math.inf

            for move, candidate, conflicts in neighbors:
                if self.is_tabu(move, self.best_conflicts, conflicts):
                    continue
                chosen_move = move
                chosen_coloring = candidate
                chosen_conflicts = conflicts
                break

            if chosen_move is None:
                # all moves tabu; allow best available
                if neighbors:
                    chosen_move, chosen_coloring, chosen_conflicts = neighbors[0]
                else:
                    break

            current = chosen_coloring
            current_conflicts = chosen_conflicts
            self.long_term_memory[chosen_move[0]] += 1

            tenure = self.adaptive_tenure(stagnation)
            self.update_tabu_list(chosen_move, tenure)

            if current_conflicts < self.best_conflicts:
                self.best_conflicts = current_conflicts
                self.best_coloring = current.copy()
                stagnation = 0
            else:
                stagnation += 1

            self.history.append({
                "iteration": self.iteration,
                "conflicts": current_conflicts,
                "best_conflicts": self.best_conflicts,
                "tabu_size": len(self.tabu_list),
            })

            if self.best_conflicts == 0:
                break
            if stagnation >= self.no_improve_limit:
                # alternate between diversification and intensification
                if self.iteration % 2 == 0:
                    current = self.diversify(current)
                else:
                    current = self.intensify(current)
                current_conflicts = self.calculate_conflicts(current)
                stagnation = 0

        self.execution_time = time.perf_counter() - start
        self.coloring = self.best_coloring
        return self.best_coloring

    def plot_convergence(self):
        if not self.history:
            return None
        iters = [h["iteration"] for h in self.history]
        conflicts = [h["conflicts"] for h in self.history]
        bests = [h["best_conflicts"] for h in self.history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, conflicts, label="Current conflicts", alpha=0.7)
        ax.plot(iters, bests, label="Best conflicts", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Conflicts")
        ax.set_title("Tabu Search Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


def tabu_search_coloring(graph: nx.Graph, max_iterations: int = 1000, tabu_tenure: int = 50) -> Dict[Any, int]:
    """Convenience wrapper for one-off tabu search coloring."""
    algo = TabuSearchColoring(graph, max_iterations=max_iterations, tabu_tenure=tabu_tenure)
    return algo.color()


__all__ = [
    "tabu_search_coloring",
    "TabuSearchColoring",
]
