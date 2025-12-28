"""
Batch experiment runner: generate many puzzles per type, run algorithms,
compare performance across families, and save results.
"""

import os
import json
import time
import argparse
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

from algorithms import (
    greedy_coloring,
    dsatur_coloring,
    welsh_powell_coloring,
    tabu_search_coloring,
    dqn_coloring,
    DQN_AVAILABLE,
)

from utils.graph_operations import (
    validate_coloring,
    count_colors,
    get_graph_stats,
)

from puzzles import (
    SudokuPuzzle,
    NQueensPuzzle,
    LatinSquarePuzzle,
    MapColoringPuzzle,
    CustomGraphPuzzle,
)

# Optional imports for specific puzzle generation
from puzzles.futoshiki import generate_futoshiki, futoshiki_to_graph
from puzzles.kakuro import generate_kakuro, kakuro_to_graph

OUTPUT_DIR = os.path.join("experiments", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGOS = {
    "greedy": greedy_coloring,
    "dsatur": dsatur_coloring,
    "welsh_powell": welsh_powell_coloring,
    "tabu_search": tabu_search_coloring,
}
if DQN_AVAILABLE:
    ALGOS["dqn"] = dqn_coloring


def run_algorithms_on_graph(graph: nx.Graph) -> List[Dict[str, Any]]:
    """Run all configured algorithms on a graph and collect metrics."""
    results = []
    for name, func in ALGOS.items():
        start = time.perf_counter()
        coloring = func(graph)
        exec_time = time.perf_counter() - start
        valid = validate_coloring(graph, coloring)
        colors_used = count_colors(coloring)
        results.append({
            "algorithm": name,
            "colors_used": colors_used,
            "execution_time": exec_time,
            "valid": valid,
            "coloring": coloring,
        })
    return results


def generate_puzzle_graphs(family: str, count: int, seed: int) -> List[Tuple[str, nx.Graph]]:
    """Generate a list of graphs for a puzzle family."""
    random.seed(seed)
    graphs: List[Tuple[str, nx.Graph]] = []

    if family == "sudoku":
        for i in range(count):
            puzzle = SudokuPuzzle(size=9)
            graphs.append((f"sudoku_9x9_{i}", puzzle.to_graph()))

    elif family == "nqueens":
        for i in range(count):
            n = random.choice([8, 12, 16])
            puzzle = NQueensPuzzle(n)
            graphs.append((f"nqueens_{n}_{i}", puzzle.to_graph()))

    elif family == "latin_square":
        for i in range(count):
            k = random.choice([5, 6, 7])
            puzzle = LatinSquarePuzzle(k)
            graphs.append((f"latin_{k}_{i}", puzzle.to_graph()))

    elif family == "map_coloring":
        for i in range(count):
            regions = random.choice([10, 15, 20])
            puzzle = MapColoringPuzzle.create_random_planar(num_regions=regions, seed=seed + i)
            graphs.append((f"planar_{regions}_{i}", puzzle.to_graph()))

    elif family == "futoshiki":
        for i in range(count):
            size = random.choice([5, 6, 7])
            grid, constraints = generate_futoshiki(size=size, seed=seed + i)
            graph = futoshiki_to_graph(grid, constraints)
            graphs.append((f"futoshiki_{size}_{i}", graph))

    elif family == "kakuro":
        for i in range(count):
            size = random.choice([5, 6, 7])
            board = generate_kakuro(size=size, seed=seed + i)
            graph = kakuro_to_graph(board)
            graphs.append((f"kakuro_{size}_{i}", graph))

    elif family == "custom_graph":
        for i in range(count):
            # Random graph family for stress tests
            n = random.choice([30, 40, 50])
            p = random.choice([0.05, 0.08, 0.1])
            G = nx.fast_gnp_random_graph(n, p, seed=seed + i)
            graphs.append((f"erdosrenyi_{n}_{i}", G))

    else:
        raise ValueError(f"Unknown family: {family}")

    return graphs


def run_batch(families: List[str], per_family: int, seed: int) -> pd.DataFrame:
    rows = []
    for family in families:
        graphs = generate_puzzle_graphs(family, per_family, seed)
        for name, graph in graphs:
            stats = get_graph_stats(graph)
            algo_results = run_algorithms_on_graph(graph)
            for res in algo_results:
                rows.append({
                    "family": family,
                    "instance": name,
                    "algorithm": res["algorithm"],
                    "colors_used": res["colors_used"],
                    "execution_time": res["execution_time"],
                    "valid": res["valid"],
                    "num_vertices": stats.get("num_nodes"),
                    "num_edges": stats.get("num_edges"),
                    "density": stats.get("density"),
                    "avg_degree": stats.get("avg_degree"),
                })
    df = pd.DataFrame(rows)
    return df


def save_outputs(df: pd.DataFrame, label: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"batch_{label}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Simple matplotlib plots
    try:
        plt.figure(figsize=(10, 6))
        for family in df["family"].unique():
            sub = df[df["family"] == family]
            plt.scatter(sub["execution_time"], sub["colors_used"], label=family, alpha=0.7)
        plt.xlabel("Execution time (s)")
        plt.ylabel("Colors used")
        plt.title("Algorithm performance across families")
        plt.legend()
        plot_path = os.path.join(OUTPUT_DIR, f"batch_{label}_{ts}.png")
        plt.savefig(plot_path, dpi=140)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Plot error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch compare algorithms across puzzle families")
    parser.add_argument("--families", nargs="*", default=[
        "sudoku", "nqueens", "latin_square", "map_coloring", "futoshiki", "kakuro", "custom_graph"
    ], help="Puzzle families to include")
    parser.add_argument("--per_family", type=int, default=5, help="Instances per family")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("DQN available:", DQN_AVAILABLE)
    df = run_batch(args.families, args.per_family, args.seed)
    print(df.groupby(["family", "algorithm"]) [["colors_used", "execution_time"]].mean())
    save_outputs(df, label=f"{args.per_family}_each")


if __name__ == "__main__":
    main()
