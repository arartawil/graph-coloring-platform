"""
Futoshiki puzzle generation, graph conversion, visualization, and validation.
Representation: grid (values, 0 = empty) and inequalities list of ((r1,c1),(r2,c2), op).
"""

import random
from typing import List, Tuple, Dict, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt


FutoshikiInequality = Tuple[Tuple[int, int], Tuple[int, int], str]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_futoshiki(n: int = 5, num_inequalities: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate an n x n Futoshiki puzzle with consistent inequalities.

    Returns a dict with keys: grid (solution grid), clues (partial grid), inequalities.
    """
    if n < 3:
        raise ValueError("n must be >= 3")

    # Generate a Latin square as a valid solution grid
    solution = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]

    # Shuffle rows and columns to randomize
    random.shuffle(solution)
    cols = list(range(n))
    random.shuffle(cols)
    solution = [[row[c] for c in cols] for row in solution]

    # Create clues by removing numbers
    clues = [row.copy() for row in solution]
    removals = int(0.6 * n * n)
    for _ in range(removals):
        r, c = random.randrange(n), random.randrange(n)
        clues[r][c] = 0

    # Inequalities
    max_ineq = n * (n - 1)
    num_ineq = num_inequalities if num_inequalities is not None else max(n, n // 2)
    num_ineq = min(num_ineq, max_ineq)
    inequalities: List[FutoshikiInequality] = []

    directions = [(0, 1), (1, 0)]  # right, down
    attempts = 0
    while len(inequalities) < num_ineq and attempts < num_ineq * 5:
        attempts += 1
        r, c = random.randrange(n), random.randrange(n)
        dr, dc = random.choice(directions)
        r2, c2 = r + dr, c + dc
        if r2 >= n or c2 >= n:
            continue
        if ((r, c), (r2, c2), '<') in inequalities or ((r2, c2), (r, c), '>') in inequalities:
            continue
        op = '<' if solution[r][c] < solution[r2][c2] else '>'
        inequalities.append(((r, c), (r2, c2), op))

    return {
        'solution': solution,
        'clues': clues,
        'inequalities': inequalities,
        'size': n,
    }


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def futoshiki_to_graph(puzzle: Dict[str, Any]) -> nx.DiGraph:
    """Convert futoshiki puzzle to directed constraint graph."""
    n = puzzle['size']
    G = nx.DiGraph()

    for r in range(n):
        for c in range(n):
            G.add_node((r, c), clue=puzzle['clues'][r][c])

    # Row/column all-different as undirected edges for coloring analogy
    for r in range(n):
        for c1 in range(n):
            for c2 in range(c1 + 1, n):
                G.add_edge((r, c1), (r, c2), type='row')
                G.add_edge((r, c2), (r, c1), type='row')
    for c in range(n):
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                G.add_edge((r1, c), (r2, c), type='col')
                G.add_edge((r2, c), (r1, c), type='col')

    # Inequalities as directed edges
    for (r1, c1), (r2, c2), op in puzzle['inequalities']:
        if op == '<':
            G.add_edge((r1, c1), (r2, c2), type='ineq', operator='<')
        else:
            G.add_edge((r2, c2), (r1, c1), type='ineq', operator='<')
    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_futoshiki(puzzle: Dict[str, Any], coloring: Optional[Dict[Tuple[int, int], int]] = None, title: str = "Futoshiki"):
    """Visualize futoshiki grid with inequalities. Returns matplotlib Figure."""
    n = puzzle['size']
    fig, ax = plt.subplots(figsize=(n, n))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw cells
    for r in range(n):
        for c in range(n):
            fill = '#fdfdfd'
            if coloring and (r, c) in coloring:
                fill = plt.cm.Pastel2(coloring[(r, c)] % 8)
            ax.add_patch(plt.Rectangle((c, n - r - 1), 1, 1, edgecolor='black', facecolor=fill))
            val = puzzle['clues'][r][c]
            if val:
                ax.text(c + 0.5, n - r - 0.5, str(val), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw inequalities
    for (r1, c1), (r2, c2), op in puzzle['inequalities']:
        x1, y1 = c1 + 0.5, n - r1 - 0.5
        x2, y2 = c2 + 0.5, n - r2 - 0.5
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1)
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(midx, midy, op, ha='center', va='center', fontsize=12, color='darkred')

    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_futoshiki_solution(puzzle: Dict[str, Any], solution: List[List[int]]) -> bool:
    """Check if a solution grid satisfies row/col uniqueness and inequalities."""
    n = puzzle['size']
    # Rows and columns
    for r in range(n):
        if len(set(solution[r])) != n:
            return False
    for c in range(n):
        col = [solution[r][c] for r in range(n)]
        if len(set(col)) != n:
            return False
    # Inequalities
    for (r1, c1), (r2, c2), op in puzzle['inequalities']:
        if op == '<' and not (solution[r1][c1] < solution[r2][c2]):
            return False
        if op == '>' and not (solution[r1][c1] > solution[r2][c2]):
            return False
    return True


class FutoshikiPuzzle:
    """Wrapper for futoshiki generation and graph conversion."""

    def __init__(self, size: int = 5, num_inequalities: Optional[int] = None):
        self.data = generate_futoshiki(size, num_inequalities)
        self.size = size

    def to_graph(self) -> nx.DiGraph:
        return futoshiki_to_graph(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'puzzle': self.data,
        }


__all__ = [
    'generate_futoshiki',
    'futoshiki_to_graph',
    'visualize_futoshiki',
    'validate_futoshiki_solution',
    'FutoshikiPuzzle'
]
