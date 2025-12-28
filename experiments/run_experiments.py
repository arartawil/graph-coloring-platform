# Run Experiments
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from algorithms import *
from puzzles import *
from utils.graph_operations import validate_coloring, count_colors

def run_algorithm_comparison(puzzle, algorithms_to_test=None):
    """
    Run multiple algorithms on a puzzle and compare results.
    
    Args:
        puzzle: Puzzle object with to_graph() method
        algorithms_to_test: List of algorithm names to test
        
    Returns:
        pandas DataFrame with results
    """
    if algorithms_to_test is None:
        algorithms_to_test = ['greedy', 'dsatur', 'welsh_powell', 'tabu_search']
    
    graph = puzzle.to_graph()
    results = []
    
    algorithm_map = {
        'greedy': greedy_coloring,
        'dsatur': dsatur_coloring,
        'welsh_powell': welsh_powell_coloring,
        'tabu_search': tabu_search_coloring,
        'dqn': dqn_coloring
    }
    
    for algo_name in algorithms_to_test:
        if algo_name not in algorithm_map:
            continue
        
        print(f"Running {algo_name}...")
        
        start_time = time.time()
        coloring = algorithm_map[algo_name](graph)
        end_time = time.time()
        
        execution_time = end_time - start_time
        colors_used = count_colors(coloring)
        is_valid = validate_coloring(graph, coloring)
        
        results.append({
            'algorithm': algo_name,
            'colors_used': colors_used,
            'time': execution_time,
            'valid': is_valid
        })
        
        print(f"  Colors used: {colors_used}, Time: {execution_time:.4f}s, Valid: {is_valid}")
    
    return pd.DataFrame(results)

def run_puzzle_suite():
    """Run experiments on multiple puzzles."""
    puzzles_to_test = [
        ('Sudoku', SudokuPuzzle()),
        ('N-Queens (8)', NQueensPuzzle(8)),
        ('Latin Square (5)', LatinSquarePuzzle(5)),
        ('Map Coloring USA', MapColoringPuzzle.create_usa_map()),
        ('Petersen Graph', CustomGraphPuzzle.create_petersen_graph()),
    ]
    
    all_results = []
    
    for puzzle_name, puzzle in puzzles_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {puzzle_name}")
        print('='*60)
        
        results = run_algorithm_comparison(puzzle)
        results['puzzle'] = puzzle_name
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    output_file = 'experiment_results.csv'
    combined_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return combined_results

if __name__ == "__main__":
    print("Starting Graph Coloring Experiments...")
    results = run_puzzle_suite()
    print("\nExperiments complete!")
    print("\nSummary:")
    print(results.groupby('algorithm')[['colors_used', 'time']].mean())
