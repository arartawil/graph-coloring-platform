"""
Sudoku Puzzle Module for Graph Coloring Platform
Provides Sudoku puzzle generation, graph conversion, visualization, and analysis.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Optional, Dict, Tuple, List, Any
import copy


class SudokuPuzzle:
    """Sudoku puzzle represented as a graph coloring problem."""
    
    def __init__(self, initial_grid=None, size=9):
        """
        Initialize Sudoku puzzle.
        
        Args:
            initial_grid: size×size numpy array with initial values (0 for empty)
            size: Size of the Sudoku (9 or 16)
        """
        if size not in [9, 16]:
            raise ValueError("Sudoku size must be 9 or 16")
        
        self.size = size
        self.box_size = int(np.sqrt(size))  # 3 for 9×9, 4 for 16×16
        self.initial_grid = initial_grid if initial_grid is not None else np.zeros((size, size), dtype=int)
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create graph representation of Sudoku."""
        G = nx.Graph()
        
        # Add nodes for each cell with initial value attribute
        for i in range(self.size):
            for j in range(self.size):
                initial_value = self.initial_grid[i, j]
                G.add_node((i, j), row=i, col=j, initial=initial_value, 
                          box=self._get_box_index(i, j))
        
        # Add edges for row constraints
        for i in range(self.size):
            for j in range(self.size):
                for k in range(j + 1, self.size):
                    G.add_edge((i, j), (i, k), constraint='row')
        
        # Add edges for column constraints
        for j in range(self.size):
            for i in range(self.size):
                for k in range(i + 1, self.size):
                    G.add_edge((i, j), (k, j), constraint='column')
        
        # Add edges for box constraints
        for box_i in range(self.box_size):
            for box_j in range(self.box_size):
                cells = [(i, j) for i in range(box_i * self.box_size, (box_i + 1) * self.box_size)
                         for j in range(box_j * self.box_size, (box_j + 1) * self.box_size)]
                for idx, cell1 in enumerate(cells):
                    for cell2 in cells[idx + 1:]:
                        if not G.has_edge(cell1, cell2):  # Avoid duplicate edges
                            G.add_edge(cell1, cell2, constraint='box')
        
        return G
    
    def _get_box_index(self, row: int, col: int) -> int:
        """Get the box index for a cell."""
        return (row // self.box_size) * self.box_size + (col // self.box_size)
    
    def to_graph(self):
        """Return the graph representation."""
        return self.graph
    
    def coloring_to_grid(self, coloring: Dict[Tuple[int, int], int]) -> np.ndarray:
        """
        Convert graph coloring to Sudoku grid.
        
        Args:
            coloring: Dictionary mapping nodes to colors
            
        Returns:
            size×size numpy array
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        for (i, j), color in coloring.items():
            grid[i, j] = color + 1  # Colors are 0-based, Sudoku is 1-based
        return grid


# ============================================================================
# Sudoku Generation Functions
# ============================================================================

def generate_sudoku(size: int = 9, difficulty: str = 'medium') -> np.ndarray:
    """
    Generate a valid Sudoku puzzle with specified difficulty.
    
    Args:
        size: Size of Sudoku (9 or 16)
        difficulty: 'easy', 'medium', 'hard', or 'expert'
            - easy: 40-45 clues (9×9) or 140-150 (16×16)
            - medium: 30-35 clues (9×9) or 110-120 (16×16)
            - hard: 25-28 clues (9×9) or 90-100 (16×16)
            - expert: 22-24 clues (9×9) or 80-85 (16×16)
    
    Returns:
        size×size numpy array with some cells filled (0 for empty)
    
    Example:
        >>> puzzle = generate_sudoku(size=9, difficulty='easy')
        >>> print(puzzle.shape)
        (9, 9)
        >>> print(np.count_nonzero(puzzle))  # Number of clues
        42
    """
    if size not in [9, 16]:
        raise ValueError("Size must be 9 or 16")
    
    # Generate a complete valid Sudoku
    complete_grid = _generate_complete_sudoku(size)
    
    # Remove numbers based on difficulty
    clue_ranges = {
        9: {
            'easy': (40, 45),
            'medium': (30, 35),
            'hard': (25, 28),
            'expert': (22, 24)
        },
        16: {
            'easy': (140, 150),
            'medium': (110, 120),
            'hard': (90, 100),
            'expert': (80, 85)
        }
    }
    
    if difficulty not in clue_ranges[size]:
        difficulty = 'medium'
    
    min_clues, max_clues = clue_ranges[size][difficulty]
    target_clues = random.randint(min_clues, max_clues)
    
    puzzle = _remove_numbers(complete_grid, size * size - target_clues)
    
    return puzzle


def _generate_complete_sudoku(size: int) -> np.ndarray:
    """
    Generate a complete valid Sudoku grid using backtracking.
    
    Args:
        size: Size of Sudoku (9 or 16)
    
    Returns:
        Complete size×size numpy array
    """
    grid = np.zeros((size, size), dtype=int)
    box_size = int(np.sqrt(size))
    
    def is_valid(grid, row, col, num):
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check box
        box_row, box_col = (row // box_size) * box_size, (col // box_size) * box_size
        if num in grid[box_row:box_row + box_size, box_col:box_col + box_size]:
            return False
        
        return True
    
    def solve(grid):
        """Solve Sudoku using backtracking with randomization."""
        # Find empty cell
        for i in range(size):
            for j in range(size):
                if grid[i, j] == 0:
                    # Try numbers in random order
                    numbers = list(range(1, size + 1))
                    random.shuffle(numbers)
                    
                    for num in numbers:
                        if is_valid(grid, i, j, num):
                            grid[i, j] = num
                            
                            if solve(grid):
                                return True
                            
                            grid[i, j] = 0
                    
                    return False
        return True
    
    solve(grid)
    return grid


def _remove_numbers(grid: np.ndarray, count: int) -> np.ndarray:
    """
    Remove numbers from a complete Sudoku grid to create a puzzle.
    
    Args:
        grid: Complete Sudoku grid
        count: Number of cells to remove
    
    Returns:
        Puzzle grid with some cells empty (0)
    """
    puzzle = grid.copy()
    size = len(grid)
    
    # Get all cell positions
    positions = [(i, j) for i in range(size) for j in range(size)]
    random.shuffle(positions)
    
    removed = 0
    for i, j in positions:
        if removed >= count:
            break
        
        # Store the value
        backup = puzzle[i, j]
        puzzle[i, j] = 0
        removed += 1
    
    return puzzle


# ============================================================================
# Graph Conversion Functions
# ============================================================================

def sudoku_to_graph(puzzle: np.ndarray) -> nx.Graph:
    """
    Convert a Sudoku puzzle to its graph representation.
    
    Args:
        puzzle: size×size numpy array (0 for empty cells)
    
    Returns:
        NetworkX graph where:
        - Nodes represent cells with attributes (row, col, value, is_clue)
        - Edges connect cells that must have different values
        - Edge attributes indicate constraint type (row/column/box)
    
    Example:
        >>> puzzle = generate_sudoku(9, 'easy')
        >>> graph = sudoku_to_graph(puzzle)
        >>> print(graph.number_of_nodes())
        81
        >>> print(graph.number_of_edges())
        810
    """
    size = len(puzzle)
    box_size = int(np.sqrt(size))
    
    G = nx.Graph()
    
    # Add nodes with attributes
    for i in range(size):
        for j in range(size):
            value = puzzle[i, j]
            G.add_node(
                (i, j),
                row=i,
                col=j,
                value=value,
                is_clue=(value != 0),
                box=(i // box_size) * box_size + (j // box_size),
                position=(i, j)
            )
    
    # Add edges for row constraints
    for i in range(size):
        for j in range(size):
            for k in range(j + 1, size):
                G.add_edge((i, j), (i, k), constraint='row', constraint_type='row')
    
    # Add edges for column constraints
    for j in range(size):
        for i in range(size):
            for k in range(i + 1, size):
                if not G.has_edge((i, j), (k, j)):
                    G.add_edge((i, j), (k, j), constraint='column', constraint_type='column')
    
    # Add edges for box constraints
    for box_i in range(box_size):
        for box_j in range(box_size):
            cells = [
                (i, j) 
                for i in range(box_i * box_size, (box_i + 1) * box_size)
                for j in range(box_j * box_size, (box_j + 1) * box_size)
            ]
            for idx, cell1 in enumerate(cells):
                for cell2 in cells[idx + 1:]:
                    if not G.has_edge(cell1, cell2):
                        G.add_edge(cell1, cell2, constraint='box', constraint_type='box')
    
    return G


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_sudoku(
    puzzle: np.ndarray, 
    coloring: Optional[Dict[Tuple[int, int], int]] = None,
    title: str = "Sudoku Puzzle"
) -> plt.Figure:
    """
    Visualize a Sudoku puzzle with optional coloring.
    
    Args:
        puzzle: size×size numpy array (0 for empty)
        coloring: Optional dictionary mapping (row, col) to color index
        title: Title for the plot
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> puzzle = generate_sudoku(9, 'easy')
        >>> fig = visualize_sudoku(puzzle, title="Easy Sudoku")
        >>> plt.show()
    """
    size = len(puzzle)
    box_size = int(np.sqrt(size))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid lines
    for i in range(size + 1):
        linewidth = 3 if i % box_size == 0 else 1
        ax.plot([0, size], [i, i], 'k-', linewidth=linewidth)
        ax.plot([i, i], [0, size], 'k-', linewidth=linewidth)
    
    # Fill cells with colors if coloring is provided
    if coloring:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Get unique colors
        color_values = list(set(coloring.values()))
        num_colors = len(color_values)
        
        # Create colormap
        cmap = cm.get_cmap('rainbow', num_colors)
        
        # Color each cell
        for (i, j), color_idx in coloring.items():
            color = cmap(color_values.index(color_idx) / max(num_colors - 1, 1))
            rect = plt.Rectangle((j, size - i - 1), 1, 1, 
                                facecolor=color, alpha=0.3, edgecolor='none')
            ax.add_patch(rect)
    
    # Fill in numbers
    for i in range(size):
        for j in range(size):
            if puzzle[i, j] != 0:
                # Clue numbers in bold
                fontweight = 'bold' if puzzle[i, j] != 0 else 'normal'
                fontsize = 20 if size == 9 else 12
                
                ax.text(j + 0.5, size - i - 0.5, str(puzzle[i, j]),
                       ha='center', va='center', 
                       fontsize=fontsize, fontweight=fontweight)
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend if coloring is provided
    if coloring:
        ax.text(size + 0.5, size - 0.5, f"Colors used: {len(color_values)}", 
               fontsize=12, va='top')
    
    plt.tight_layout()
    return fig


def visualize_sudoku_simple(puzzle: np.ndarray, title: str = "Sudoku") -> None:
    """
    Simple console visualization of Sudoku puzzle.
    
    Args:
        puzzle: size×size numpy array
        title: Title to print
    
    Example:
        >>> puzzle = generate_sudoku(9, 'easy')
        >>> visualize_sudoku_simple(puzzle)
    """
    size = len(puzzle)
    box_size = int(np.sqrt(size))
    
    print(f"\n{title}")
    print("=" * (size * 2 + box_size + 1))
    
    for i in range(size):
        if i > 0 and i % box_size == 0:
            print("-" * (size * 2 + box_size + 1))
        
        row = ""
        for j in range(size):
            if j > 0 and j % box_size == 0:
                row += "| "
            
            if puzzle[i, j] == 0:
                row += ". "
            else:
                row += f"{puzzle[i, j]} "
        
        print(row)
    
    print("=" * (size * 2 + box_size + 1))


# ============================================================================
# Graph Property Functions
# ============================================================================

def get_sudoku_properties(graph: nx.Graph) -> Dict[str, Any]:
    """
    Calculate graph properties of a Sudoku puzzle.
    
    Args:
        graph: NetworkX graph representing Sudoku
    
    Returns:
        Dictionary containing:
        - num_vertices: Number of cells
        - num_edges: Number of constraints
        - density: Graph density
        - avg_degree: Average node degree
        - max_degree: Maximum node degree
        - chromatic_number_lower: Lower bound on chromatic number
        - chromatic_number_upper: Upper bound on chromatic number
        - is_regular: Whether graph is regular
        - num_clues: Number of pre-filled cells
        - clique_number: Size of maximum clique
    
    Example:
        >>> puzzle = generate_sudoku(9, 'medium')
        >>> graph = sudoku_to_graph(puzzle)
        >>> props = get_sudoku_properties(graph)
        >>> print(f"Chromatic number: {props['chromatic_number_lower']}-{props['chromatic_number_upper']}")
    """
    size = int(np.sqrt(graph.number_of_nodes()))
    
    # Basic graph properties
    num_vertices = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph)
    
    # Degree statistics
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())
    avg_degree = np.mean(degree_values)
    max_degree = max(degree_values)
    min_degree = min(degree_values)
    is_regular = (max_degree == min_degree)
    
    # Chromatic number bounds
    # Lower bound: maximum clique size
    # For Sudoku, each row/column/box forms a clique of size 'size'
    chromatic_number_lower = size
    
    # Upper bound: maximum degree + 1 (greedy bound)
    chromatic_number_upper = max_degree + 1
    
    # Actual chromatic number for Sudoku is exactly 'size'
    chromatic_number_exact = size
    
    # Count clues (pre-filled cells)
    num_clues = sum(1 for node in graph.nodes() if graph.nodes[node].get('is_clue', False))
    
    # Clique number (size of maximum clique)
    # For Sudoku, it's the size of row/column/box
    clique_number = size
    
    # Additional properties
    is_connected = nx.is_connected(graph)
    num_triangles = sum(nx.triangles(graph).values()) // 3
    clustering_coefficient = nx.average_clustering(graph)
    
    properties = {
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'density': round(density, 4),
        'avg_degree': round(avg_degree, 2),
        'max_degree': max_degree,
        'min_degree': min_degree,
        'is_regular': is_regular,
        'chromatic_number_lower': chromatic_number_lower,
        'chromatic_number_upper': chromatic_number_upper,
        'chromatic_number_exact': chromatic_number_exact,
        'num_clues': num_clues,
        'clique_number': clique_number,
        'is_connected': is_connected,
        'num_triangles': num_triangles,
        'clustering_coefficient': round(clustering_coefficient, 4),
        'sudoku_size': size
    }
    
    return properties


def analyze_sudoku_difficulty(puzzle: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the difficulty of a Sudoku puzzle.
    
    Args:
        puzzle: size×size numpy array
    
    Returns:
        Dictionary with difficulty metrics:
        - num_clues: Number of given clues
        - empty_cells: Number of empty cells
        - difficulty_score: Estimated difficulty (0-100)
        - difficulty_level: Estimated level (easy/medium/hard/expert)
    
    Example:
        >>> puzzle = generate_sudoku(9, 'hard')
        >>> analysis = analyze_sudoku_difficulty(puzzle)
        >>> print(analysis['difficulty_level'])
        'hard'
    """
    size = len(puzzle)
    total_cells = size * size
    num_clues = np.count_nonzero(puzzle)
    empty_cells = total_cells - num_clues
    
    # Calculate difficulty score based on number of clues
    clue_ratio = num_clues / total_cells
    
    # More clues = easier (inverse relationship)
    difficulty_score = int((1 - clue_ratio) * 100)
    
    # Classify difficulty
    if size == 9:
        if num_clues >= 40:
            difficulty_level = 'easy'
        elif num_clues >= 30:
            difficulty_level = 'medium'
        elif num_clues >= 25:
            difficulty_level = 'hard'
        else:
            difficulty_level = 'expert'
    else:  # 16×16
        if num_clues >= 140:
            difficulty_level = 'easy'
        elif num_clues >= 110:
            difficulty_level = 'medium'
        elif num_clues >= 90:
            difficulty_level = 'hard'
        else:
            difficulty_level = 'expert'
    
    return {
        'num_clues': num_clues,
        'empty_cells': empty_cells,
        'clue_ratio': round(clue_ratio, 3),
        'difficulty_score': difficulty_score,
        'difficulty_level': difficulty_level,
        'size': size
    }


# ============================================================================
# Utility Functions
# ============================================================================

def is_valid_sudoku(grid: np.ndarray) -> bool:
    """
    Check if a Sudoku solution is valid.
    
    Args:
        grid: Complete size×size numpy array
    
    Returns:
        True if valid, False otherwise
    """
    size = len(grid)
    box_size = int(np.sqrt(size))
    
    # Check rows
    for i in range(size):
        if len(set(grid[i, :])) != size or 0 in grid[i, :]:
            return False
    
    # Check columns
    for j in range(size):
        if len(set(grid[:, j])) != size or 0 in grid[:, j]:
            return False
    
    # Check boxes
    for box_i in range(box_size):
        for box_j in range(box_size):
            box = grid[box_i * box_size:(box_i + 1) * box_size,
                      box_j * box_size:(box_j + 1) * box_size]
            if len(set(box.flatten())) != size or 0 in box.flatten():
                return False
    
    return True


def solve_sudoku(puzzle: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve a Sudoku puzzle using backtracking.
    
    Args:
        puzzle: size×size numpy array (0 for empty)
    
    Returns:
        Solved grid or None if unsolvable
    """
    grid = puzzle.copy()
    size = len(grid)
    box_size = int(np.sqrt(size))
    
    def is_valid(grid, row, col, num):
        # Check row
        if num in grid[row]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check box
        box_row = (row // box_size) * box_size
        box_col = (col // box_size) * box_size
        if num in grid[box_row:box_row + box_size, box_col:box_col + box_size]:
            return False
        
        return True
    
    def solve(grid):
        for i in range(size):
            for j in range(size):
                if grid[i, j] == 0:
                    for num in range(1, size + 1):
                        if is_valid(grid, i, j, num):
                            grid[i, j] = num
                            
                            if solve(grid):
                                return True
                            
                            grid[i, j] = 0
                    
                    return False
        return True
    
    if solve(grid):
        return grid
    return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Generate and visualize a Sudoku puzzle
    print("Example 1: Generate Sudoku")
    puzzle = generate_sudoku(size=9, difficulty='medium')
    print(f"Generated {np.count_nonzero(puzzle)} clues")
    visualize_sudoku_simple(puzzle, "Medium Sudoku Puzzle")
    
    # Example 2: Convert to graph and analyze
    print("\nExample 2: Graph Analysis")
    graph = sudoku_to_graph(puzzle)
    props = get_sudoku_properties(graph)
    print(f"Vertices: {props['num_vertices']}")
    print(f"Edges: {props['num_edges']}")
    print(f"Chromatic number: {props['chromatic_number_exact']}")
    print(f"Average degree: {props['avg_degree']}")
    
    # Example 3: Difficulty analysis
    print("\nExample 3: Difficulty Analysis")
    analysis = analyze_sudoku_difficulty(puzzle)
    print(f"Difficulty: {analysis['difficulty_level']}")
    print(f"Clues: {analysis['num_clues']}")
    print(f"Score: {analysis['difficulty_score']}/100")
