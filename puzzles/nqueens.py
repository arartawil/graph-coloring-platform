"""
N-Queens puzzle generation, graph conversion, visualization, and solving utilities.
Supports large boards (8, 16, 32, 64, 100) with conflict detection.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Optional
import random


class NQueensPuzzle:
    """N-Queens puzzle wrapper providing a graph representation for coloring."""

    def __init__(self, n: int = 8):
        if n < 4:
            raise ValueError("n must be at least 4")
        self.n = n
        # A solved board is stored for reference/visualization; coloring uses the conflict graph.
        self.solution = generate_nqueens(n)
        self.graph = nqueens_to_graph(n)
        # Use board coordinates as positions for nicer plotting.
        positions = {(r, c): (c, -r) for r in range(n) for c in range(n)}
        nx.set_node_attributes(self.graph, positions, "pos")

    def to_graph(self) -> nx.Graph:
        """Return the conflict graph for this N-Queens instance."""
        return self.graph

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_nqueens(n: int = 8) -> List[int]:
    """
    Generate a (ideally conflict-free) N-Queens board configuration.

    Supports general n >= 4, using constructive patterns when available,
    backtracking for moderate n, and min-conflicts for larger boards.
    """
    if n < 4:
        raise ValueError("n must be at least 4")

    # Constructive pattern for many even n values (no conflicts)
    if n % 2 == 0 and n % 6 not in (2, 3):
        evens = list(range(2, n + 1, 2))
        odds = list(range(1, n + 1, 2))
        return [c - 1 for c in evens + odds]

    # For modest sizes, use exact backtracking; otherwise heuristic
    if n <= 16:
        return solve_nqueens_backtracking(n)
    return heuristic_nqueens(n)


def heuristic_nqueens(n: int, max_iter: int = 20000) -> List[int]:
    """
    Min-conflicts heuristic for large N-Queens.
    Returns a conflict-free configuration or best found.
    """
    board = [random.randrange(n) for _ in range(n)]

    def conflicts(row: int, col: int) -> int:
        count = 0
        for r, c in enumerate(board):
            if r == row:
                continue
            if c == col or abs(r - row) == abs(c - col):
                count += 1
        return count

    for _ in range(max_iter):
        conflict_rows = [r for r in range(n) if conflicts(r, board[r]) > 0]
        if not conflict_rows:
            return board
        row = random.choice(conflict_rows)
        best_col = min(range(n), key=lambda c: conflicts(row, c))
        board[row] = best_col
    return board


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------

def nqueens_to_graph(n: int) -> nx.Graph:
    """
    Convert N-Queens to a conflict graph.

    Vertices: each square (row, col)
    Edges: attacking positions (same row, same column, diagonals)

    Args:
        n: Board size

    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    for r in range(n):
        for c in range(n):
            G.add_node((r, c), row=r, col=c)

    for r in range(n):
        for c in range(n):
            for k in range(n):
                if k != c:
                    G.add_edge((r, c), (r, k), constraint="row")
                if k != r:
                    G.add_edge((r, c), (k, c), constraint="col")
            # Diagonals
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                rr, cc = r + dr, c + dc
                while 0 <= rr < n and 0 <= cc < n:
                    G.add_edge((r, c), (rr, cc), constraint="diag")
                    rr += dr
                    cc += dc
    return G


# ---------------------------------------------------------------------------
# Conflict utilities
# ---------------------------------------------------------------------------

def detect_conflicts(board: List[int]) -> List[Tuple[int, int]]:
    """Return list of conflicting row pairs given a board configuration."""
    conflicts = []
    n = len(board)
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            c1, c2 = board[r1], board[r2]
            if c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                conflicts.append((r1, r2))
    return conflicts


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_nqueens(board: List[int], coloring: Optional[Dict[Tuple[int, int], int]] = None) -> go.Figure:
    """
    Visualize N-Queens board with Plotly.

    Args:
        board: list where index=row and value=col for queen positions
        coloring: optional mapping of (row, col) -> color index for conflicts

    Returns:
        Plotly Figure
    """
    n = len(board)
    # Board squares
    squares_x, squares_y, square_colors = [], [], []
    for r in range(n):
        for c in range(n):
            squares_x += [c, c + 1, c + 1, c, None]
            squares_y += [r, r, r + 1, r + 1, None]
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            square_colors.append(color)

    board_color = []
    # Plotly scatter for queens
    queen_x = [c + 0.5 for c in board]
    queen_y = [r + 0.5 for r in range(n)]

    queen_colors = []
    for r, c in enumerate(board):
        if coloring and (r, c) in coloring:
            queen_colors.append(coloring[(r, c)])
        else:
            queen_colors.append(0)

    fig = go.Figure()
    # Add squares as shapes
    for r in range(n):
        for c in range(n):
            color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
            fig.add_shape(type="rect", x0=c, y0=r, x1=c + 1, y1=r + 1, line=dict(width=0), fillcolor=color)

    fig.add_trace(go.Scatter(
        x=queen_x,
        y=queen_y,
        mode="markers",
        marker=dict(
            size=32,
            symbol="queen",
            color=queen_colors,
            colorscale="Turbo",
            showscale=bool(coloring)
        ),
        hovertext=[f"Row {r}, Col {c}" for r, c in enumerate(board)],
        hoverinfo="text"
    ))

    fig.update_layout(
        width=650,
        height=650,
        xaxis=dict(range=[0, n], showgrid=False, zeroline=False, tickmode="array", tickvals=list(range(n)), ticktext=list(range(n))),
        yaxis=dict(range=[0, n], showgrid=False, zeroline=False, tickmode="array", tickvals=list(range(n)), ticktext=list(range(n))),
        title=f"N-Queens ({n}x{n})",
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve_nqueens_backtracking(n: int = 8) -> List[int]:
    """
    Solve N-Queens via backtracking. Suitable for n <= 16; for larger n falls back to heuristic.

    Args:
        n: Board size

    Returns:
        Conflict-free board configuration
    """
    if n > 16:
        return heuristic_nqueens(n)

    board = [-1] * n

    def is_safe(row: int, col: int) -> bool:
        for r in range(row):
            c = board[r]
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                if backtrack(row + 1):
                    return True
                board[row] = -1
        return False

    backtrack(0)
    return board


# ---------------------------------------------------------------------------
# Example usage (manual test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = 8
    board = solve_nqueens_backtracking(n)
    print(f"Solution for {n}-Queens: {board}")
    print(f"Conflicts: {detect_conflicts(board)}")
