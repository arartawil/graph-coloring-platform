"""
Graph visualization with coloring results.
Exports individual colored graphs as images for research paper.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import puzzle generators
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import greedy_coloring, dsatur_coloring, welsh_powell_coloring, smallest_last_coloring, largest_first_coloring, iterated_greedy_coloring
from puzzles import (
    SudokuPuzzle, NQueensPuzzle, LatinSquarePuzzle, 
    MapColoringPuzzle, CustomGraphPuzzle
)
from puzzles.sudoku import generate_sudoku, sudoku_to_graph
from puzzles.latin_square import generate_latin_square, latin_square_to_graph
from puzzles.futoshiki import generate_futoshiki, futoshiki_to_graph
from puzzles.kakuro import generate_kakuro, kakuro_to_graph

RESULTS_DIR = "results"
GRAPHS_DIR = os.path.join(RESULTS_DIR, "colored_graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

ALGO_COLORS = {
    'greedy': '#FF6B6B',
    'dsatur': '#4ECDC4',
    'welsh_powell': '#45B7D1'
}

CMAP = plt.cm.get_cmap('tab20')


def create_sudoku_graph():
    """Generate and return a sudoku puzzle graph."""
    puzzle = generate_sudoku(size=9, difficulty='medium')
    graph = sudoku_to_graph(puzzle)
    return graph, "sudoku_9x9_medium"


def create_nqueens_graph():
    """Generate and return an N-Queens puzzle graph."""
    puzzle = NQueensPuzzle(8)
    return puzzle.to_graph(), "nqueens_8"


def create_latin_square_graph():
    """Generate and return a Latin square graph."""
    n = 5
    generate_latin_square(n, clue_fraction=0.5)
    graph = latin_square_to_graph(n)
    return graph, "latin_square_5"


def create_map_coloring_graph():
    """Generate and return a map coloring graph."""
    puzzle = MapColoringPuzzle.create_usa_map()
    return puzzle.to_graph(), "map_coloring_usa"


def create_futoshiki_graph():
    """Generate and return a Futoshiki graph."""
    puzzle = generate_futoshiki(n=5)
    graph = futoshiki_to_graph(puzzle)
    return graph, "futoshiki_5x5"


def create_custom_random_graph():
    """Generate and return a random Erdős–Rényi graph."""
    graph = nx.erdos_renyi_graph(20, 0.15, seed=42)
    return graph, "random_erdos_renyi_20"


def visualize_colored_graph(graph, coloring, title, filename, algo_name=''):
    """Visualize a graph with node coloring."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # Extract colors for each node
    node_colors = [coloring.get(node, 0) for node in graph.nodes()]
    
    # Draw network
    nx.draw_networkx_edges(
        graph, pos,
        ax=ax,
        edge_color='gray',
        alpha=0.3,
        width=0.5
    )
    
    nodes = nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=300,
        cmap='tab20',
        vmin=0,
        vmax=19,
        ax=ax,
        edgecolors='black',
        linewidths=1.5
    )
    
    # Add labels only for small graphs
    if len(graph.nodes()) <= 30:
        nx.draw_networkx_labels(
            graph, pos,
            ax=ax,
            font_size=8,
            font_weight='bold'
        )
    
    num_colors = len(set(node_colors))
    ax.set_title(
        f'{title}\n{algo_name} | Colors Used: {num_colors}',
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return num_colors


def visualize_comparison(graph, colorings_dict, title, filename):
    """Visualize graph with multiple algorithm colorings side-by-side."""
    num_algos = len(colorings_dict)
    fig, axes = plt.subplots(1, num_algos, figsize=(6*num_algos, 6))
    
    if num_algos == 1:
        axes = [axes]
    
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    for idx, (algo_name, coloring) in enumerate(colorings_dict.items()):
        ax = axes[idx]
        node_colors = [coloring.get(node, 0) for node in graph.nodes()]
        num_colors = len(set(node_colors))
        
        nx.draw_networkx_edges(
            graph, pos,
            ax=ax,
            edge_color='gray',
            alpha=0.3,
            width=0.5
        )
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=250,
            cmap='tab20',
            vmin=0,
            vmax=19,
            ax=ax,
            edgecolors='black',
            linewidths=1
        )
        
        if len(graph.nodes()) <= 20:
            nx.draw_networkx_labels(
                graph, pos,
                ax=ax,
                font_size=7,
                font_weight='bold'
            )
        
        ax.set_title(
            f'{algo_name}\nColors: {num_colors}',
            fontsize=11,
            fontweight='bold'
        )
        ax.axis('off')
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    """Generate colored graph visualizations."""
    print("\n🎨 Generating colored graph visualizations...\n")
    
    # Define test cases
    test_cases = [
        ("Sudoku (9×9)", create_sudoku_graph),
        ("N-Queens (n=8)", create_nqueens_graph),
        ("Latin Square (5×5)", create_latin_square_graph),
        ("Map Coloring (USA)", create_map_coloring_graph),
        ("Futoshiki (5×5)", create_futoshiki_graph),
        ("Random Graph (Erdős–Rényi)", create_custom_random_graph),
    ]
    
    algorithms = {
        'Greedy': greedy_coloring,
        'DSatur': dsatur_coloring,
        'Welsh-Powell': welsh_powell_coloring,
        'Smallest Last': smallest_last_coloring,
        'Largest First': largest_first_coloring,
        'Iterated Greedy': iterated_greedy_coloring,
    }
    
    results = []
    
    for puzzle_name, graph_builder in test_cases:
        print(f"Processing: {puzzle_name}")
        try:
            graph, graph_id = graph_builder()
            
            # Run all algorithms
            colorings = {}
            colors_used = {}
            
            for algo_name, algo_func in algorithms.items():
                coloring = algo_func(graph)
                colorings[algo_name] = coloring
                colors_used[algo_name] = len(set(coloring.values()))
            
            # Save comparison visualization
            comparison_file = os.path.join(
                GRAPHS_DIR,
                f"{graph_id}_comparison.png"
            )
            visualize_comparison(graph, colorings, puzzle_name, comparison_file)
            print(f"  ✓ Saved: {comparison_file}")
            
            # Save individual visualizations
            for algo_name, coloring in colorings.items():
                individual_file = os.path.join(
                    GRAPHS_DIR,
                    f"{graph_id}_{algo_name.lower().replace('-', '_')}.png"
                )
                num_colors = visualize_colored_graph(
                    graph, coloring, puzzle_name, individual_file, algo_name
                )
                print(f"    • {algo_name}: {individual_file}")
            
            # Store results
            for algo_name, num_colors in colors_used.items():
                results.append({
                    'puzzle': puzzle_name,
                    'graph_id': graph_id,
                    'nodes': len(graph.nodes()),
                    'edges': len(graph.edges()),
                    'algorithm': algo_name,
                    'colors_used': num_colors
                })
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Save results summary
    results_df = pd.DataFrame(results)
    summary_file = os.path.join(RESULTS_DIR, "colored_graphs_summary.csv")
    results_df.to_csv(summary_file, index=False)
    
    print(f"✅ All visualizations saved to: {GRAPHS_DIR}/")
    print(f"✅ Summary saved to: {summary_file}\n")
    print("Generated files:")
    print("  *_comparison.png  — Side-by-side algorithm comparison")
    print("  *_greedy.png      — Greedy coloring result")
    print("  *_dsatur.png      — DSatur coloring result")
    print("  *_welsh_powell.png — Welsh-Powell coloring result\n")
    
    print("Summary of results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
