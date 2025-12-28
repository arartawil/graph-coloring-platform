"""
Latin Square generation, graph conversion, visualization.
Simple version: no sub-box constraints (unlike Sudoku).
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_latin_square(n: int = 5, clue_fraction: float = 0.5) -> np.ndarray:
    """
    Generate an n x n Latin square and remove entries to form a puzzle.
    Uses a cyclic Latin square for simplicity.
    """
    base = np.array([[(i + j) % n + 1 for j in range(n)] for i in range(n)])
    puzzle = base.copy()

    # Remove clues according to fraction
    total = n * n
    remove_count = int(total * (1 - clue_fraction))
    indices = [(i, j) for i in range(n) for j in range(n)]
    np.random.shuffle(indices)
    for i, j in indices[:remove_count]:
        puzzle[i, j] = 0
    return puzzle


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def latin_square_to_graph(n: int) -> nx.Graph:
    """
    Convert an n x n Latin square template to a graph where cells in the same row
    or column are connected (all-different constraints).
    """
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j))
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                G.add_edge((i, j), (i, k), constraint='row')
    for j in range(n):
        for i in range(n):
            for k in range(i + 1, n):
                G.add_edge((i, j), (k, j), constraint='col')
    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_latin_square(square: np.ndarray, coloring: Optional[Dict[Tuple[int, int], int]] = None, title: str = "Latin Square"):
    """Visualize a Latin square with optional coloring. Returns matplotlib Figure."""
    n = square.shape[0]
    fig, ax = plt.subplots(figsize=(n, n))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(n):
        for j in range(n):
            fill = '#ffffff'
            if coloring and (i, j) in coloring:
                fill = plt.cm.Pastel1(coloring[(i, j)] % 9)
            ax.add_patch(plt.Rectangle((j, n - i - 1), 1, 1, edgecolor='black', facecolor=fill))
            if square[i, j] != 0:
                ax.text(j + 0.5, n - i - 0.5, str(square[i, j]), ha='center', va='center', fontsize=12)

    ax.set_title(title)
    return fig


class LatinSquarePuzzle:
    """Lightweight wrapper around Latin square generation and graph conversion."""

    def __init__(self, size: int = 5, clue_fraction: float = 0.5):
        self.size = size
        self.clue_fraction = clue_fraction
        self.square = generate_latin_square(size, clue_fraction)

    def to_graph(self) -> nx.Graph:
        return latin_square_to_graph(self.size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'clue_fraction': self.clue_fraction,
            'square': self.square.tolist(),
        }


__all__ = [
    'generate_latin_square',
    'latin_square_to_graph',
    'visualize_latin_square',
    'LatinSquarePuzzle'
]
