"""
Kakuro puzzle generation, graph conversion, visualization, and validation.
Representation:
    - Grid of dict cells. Black cells have clue_down/clue_across; white cells have value (0 if empty).
"""

import random
from typing import Dict, Tuple, List, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt

Cell = Dict[str, Any]
KakuroGrid = List[List[Cell]]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_kakuro(rows: int = 10, cols: int = 10) -> KakuroGrid:
    """
    Generate a Kakuro grid with black clue cells and white fill cells.

    The generator builds horizontal and vertical runs of length 2-5 and assigns
    consistent clue sums using unique digits 1-9 per run.
    """
    if rows < 5 or cols < 5:
        raise ValueError("Kakuro grid should be at least 5x5")

    grid: KakuroGrid = [[{'type': 'white', 'value': 0} for _ in range(cols)] for _ in range(rows)]

    # Place black cells at borders for simplicity
    for r in range(rows):
        for c in range(cols):
            if r == 0 or c == 0:
                grid[r][c] = {'type': 'black', 'clue_across': None, 'clue_down': None}

    # Helper to create a run of white cells and assign a clue
    def create_run(start: Tuple[int, int], direction: Tuple[int, int], length: int):
        cells = []
        r, c = start
        for _ in range(length):
            r += direction[0]
            c += direction[1]
            if r >= rows or c >= cols:
                return None
            if grid[r][c]['type'] == 'black':
                return None
            cells.append((r, c))
        return cells

    def assign_clue_for_run(clue_r: int, clue_c: int, cells: List[Tuple[int, int]], direction: str):
        length = len(cells)
        digits = random.sample(range(1, 10), length)
        clue_sum = sum(digits)
        if direction == 'across':
            grid[clue_r][clue_c]['clue_across'] = clue_sum
        else:
            grid[clue_r][clue_c]['clue_down'] = clue_sum
        # Store run membership
        for cell in cells:
            grid[cell[0]][cell[1]].setdefault('runs', []).append({'direction': direction, 'sum': clue_sum, 'cells': cells})

    # Create horizontal runs
    for r in range(1, rows):
        c = 1
        while c < cols:
            if grid[r][c]['type'] == 'black':
                c += 1
                continue
            # decide run length
            length = random.choice([2, 3, 4, 5])
            run_cells = create_run((r, c - 1), (0, 1), length)
            if run_cells:
                grid[r][c - 1] = {'type': 'black', 'clue_across': None, 'clue_down': None}
                assign_clue_for_run(r, c - 1, run_cells, 'across')
                c += length
            else:
                c += 1

    # Create vertical runs
    for c in range(1, cols):
        r = 1
        while r < rows:
            if grid[r][c]['type'] == 'black':
                r += 1
                continue
            length = random.choice([2, 3, 4, 5])
            run_cells = create_run((r - 1, c), (1, 0), length)
            if run_cells:
                grid[r - 1][c] = grid[r - 1][c] if grid[r - 1][c]['type'] == 'black' else {'type': 'black', 'clue_across': None, 'clue_down': None}
                assign_clue_for_run(r - 1, c, run_cells, 'down')
                r += length
            else:
                r += 1

    return grid


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def kakuro_to_graph(puzzle: KakuroGrid) -> nx.Graph:
    """
    Convert Kakuro puzzle to a graph where white cells are vertices and edges
    connect cells in the same run. Edge attributes carry sum constraints.
    """
    G = nx.Graph()
    rows, cols = len(puzzle), len(puzzle[0])
    # Add nodes for white cells
    for r in range(rows):
        for c in range(cols):
            cell = puzzle[r][c]
            if cell['type'] == 'white':
                G.add_node((r, c), value=cell.get('value', 0))

    # Add edges per run
    for r in range(rows):
        for c in range(cols):
            cell = puzzle[r][c]
            if cell['type'] == 'white' and 'runs' in cell:
                for run in cell['runs']:
                    cells = run['cells']
                    for i in range(len(cells)):
                        for j in range(i + 1, len(cells)):
                            u, v = cells[i], cells[j]
                            if G.has_edge(u, v):
                                continue
                            G.add_edge(u, v, direction=run['direction'], target_sum=run['sum'], run_size=len(cells))
    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_kakuro(puzzle: KakuroGrid, coloring: Optional[Dict[Tuple[int, int], int]] = None, title: str = "Kakuro"):
    """Visualize Kakuro grid with clues and optional coloring. Returns matplotlib Figure."""
    rows, cols = len(puzzle), len(puzzle[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    for r in range(rows):
        for c in range(cols):
            cell = puzzle[r][c]
            x, y = c, rows - r - 1
            if cell['type'] == 'black':
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))
                clue_a = cell.get('clue_across')
                clue_d = cell.get('clue_down')
                if clue_a or clue_d:
                    ax.plot([x, x + 1], [y, y + 1], color='white', linewidth=1)
                    if clue_a:
                        ax.text(x + 0.75, y + 0.1, str(clue_a), color='white', ha='right', va='bottom', fontsize=8)
                    if clue_d:
                        ax.text(x + 0.1, y + 0.75, str(clue_d), color='white', ha='left', va='top', fontsize=8)
            else:
                fill = '#f6f6f6'
                if coloring and (r, c) in coloring:
                    fill = plt.cm.Pastel1(coloring[(r, c)] % 9)
                ax.add_patch(plt.Rectangle((x, y), 1, 1, edgecolor='gray', facecolor=fill))
                val = cell.get('value', 0)
                if val:
                    ax.text(x + 0.5, y + 0.5, str(val), ha='center', va='center', fontsize=10)

    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_kakuro(puzzle: KakuroGrid) -> bool:
    """Check basic Kakuro constraints: unique digits per run and correct sums."""
    # Collect runs
    seen_runs = set()
    for r in range(len(puzzle)):
        for c in range(len(puzzle[0])):
            cell = puzzle[r][c]
            if cell['type'] == 'white' and 'runs' in cell:
                for run in cell['runs']:
                    run_key = (tuple(run['cells']), run['direction'])
                    if run_key in seen_runs:
                        continue
                    seen_runs.add(run_key)
                    values = [puzzle[rr][cc].get('value', 0) for rr, cc in run['cells']]
                    non_zero = [v for v in values if v != 0]
                    if len(non_zero) != len(set(non_zero)):
                        return False
                    if all(values):
                        if sum(values) != run['sum']:
                            return False
    return True


class KakuroPuzzle:
    """Wrapper for Kakuro generation and graph conversion."""

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.grid = generate_kakuro(rows, cols)

    def to_graph(self) -> nx.Graph:
        return kakuro_to_graph(self.grid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rows': self.rows,
            'cols': self.cols,
            'grid': self.grid,
        }


__all__ = [
    'generate_kakuro',
    'kakuro_to_graph',
    'visualize_kakuro',
    'validate_kakuro',
    'KakuroPuzzle'
]
