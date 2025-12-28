"""
Research-grade experiment runner for all coloring algorithms across puzzle families
and difficulty levels. Results (CSV + figures) are written to the top-level
`results/` folder for use in papers or reports.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms import (
    greedy_coloring,
    dsatur_coloring,
    welsh_powell_coloring,
    smallest_last_coloring,
    largest_first_coloring,
    iterated_greedy_coloring,
    tabu_search_coloring,
    dqn_coloring,
    DQN_AVAILABLE,
)
from utils.graph_operations import validate_coloring, count_colors, get_graph_stats
from puzzles import (
    SudokuPuzzle,
    NQueensPuzzle,
    LatinSquarePuzzle,
    MapColoringPuzzle,
    CustomGraphPuzzle,
)
from puzzles.sudoku import generate_sudoku, sudoku_to_graph
from puzzles.latin_square import generate_latin_square, latin_square_to_graph
from puzzles.futoshiki import generate_futoshiki, futoshiki_to_graph
from puzzles.kakuro import generate_kakuro, kakuro_to_graph

RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGOS: Dict[str, Any] = {
    "greedy": greedy_coloring,
    "dsatur": dsatur_coloring,
    "welsh_powell": welsh_powell_coloring,
    "smallest_last": smallest_last_coloring,
    "largest_first": largest_first_coloring,
    "iterated_greedy": iterated_greedy_coloring,
    "tabu_search": tabu_search_coloring,
}
if DQN_AVAILABLE:
    ALGOS["dqn"] = dqn_coloring

# Default difficulty ladder per puzzle family
PUZZLE_LEVELS: Dict[str, List[Dict[str, Any]]] = {
    "sudoku": [
        {"level": "easy", "size": 9, "difficulty": "easy", "count": 2},
        {"level": "medium", "size": 9, "difficulty": "medium", "count": 2},
        {"level": "hard", "size": 9, "difficulty": "hard", "count": 2},
        {"level": "expert", "size": 9, "difficulty": "expert", "count": 2},
    ],
    "nqueens": [
        {"level": "small", "size": 8, "count": 2},
        {"level": "medium", "size": 16, "count": 2},
        {"level": "large", "size": 32, "count": 1},
    ],
    "latin_square": [
        {"level": "light", "size": 5, "clue_fraction": 0.7, "count": 2},
        {"level": "balanced", "size": 7, "clue_fraction": 0.5, "count": 2},
        {"level": "sparse", "size": 9, "clue_fraction": 0.35, "count": 1},
    ],
    "map_coloring": [
        {"level": "small", "regions": 12, "count": 2},
        {"level": "medium", "regions": 18, "count": 2},
        {"level": "large", "regions": 24, "count": 1},
    ],
    "futoshiki": [
        {"level": "small", "size": 5, "count": 2},
        {"level": "medium", "size": 6, "count": 2},
        {"level": "large", "size": 7, "count": 1},
    ],
    "kakuro": [
        {"level": "small", "rows": 8, "cols": 8, "count": 2},
        {"level": "medium", "rows": 10, "cols": 10, "count": 2},
        {"level": "large", "rows": 12, "cols": 12, "count": 1},
    ],
    "custom_graph": [
        {"level": "sparse", "nodes": 30, "edge_prob": 0.05, "count": 2},
        {"level": "medium", "nodes": 40, "edge_prob": 0.08, "count": 2},
        {"level": "dense", "nodes": 50, "edge_prob": 0.12, "count": 1},
    ],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def build_sudoku(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    puzzle = generate_sudoku(size=level_cfg.get("size", 9), difficulty=level_cfg.get("difficulty", "medium"))
    graph = sudoku_to_graph(puzzle)
    meta = {
        "size": level_cfg.get("size", 9),
        "difficulty": level_cfg.get("difficulty", "medium"),
        "given_cells": int(np.count_nonzero(puzzle)),
    }
    return graph, meta


def build_nqueens(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    n = level_cfg.get("size", 8)
    puzzle = NQueensPuzzle(n)
    return puzzle.to_graph(), {"size": n}


def build_latin_square(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    n = level_cfg.get("size", 5)
    clue_fraction = level_cfg.get("clue_fraction", 0.5)
    generate_latin_square(n, clue_fraction)
    graph = latin_square_to_graph(n)
    return graph, {"size": n, "clue_fraction": clue_fraction}


def build_map(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    regions = level_cfg.get("regions", 12)
    puzzle = MapColoringPuzzle.create_random_planar(num_regions=regions, seed=seed)
    return puzzle.to_graph(), {"regions": regions}


def build_futoshiki(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    size = level_cfg.get("size", 5)
    puzzle = generate_futoshiki(n=size)
    graph = futoshiki_to_graph(puzzle)
    return graph, {"size": size, "num_inequalities": len(puzzle.get("inequalities", []))}


def build_kakuro(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    rows = level_cfg.get("rows", 10)
    cols = level_cfg.get("cols", 10)
    puzzle = generate_kakuro(rows=rows, cols=cols)
    graph = kakuro_to_graph(puzzle)
    return graph, {"rows": rows, "cols": cols}


def build_custom_graph(level_cfg: Dict[str, Any], seed: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    seed_everything(seed)
    nodes = level_cfg.get("nodes", 30)
    prob = level_cfg.get("edge_prob", 0.08)
    graph = nx.erdos_renyi_graph(nodes, prob, seed=seed)
    return graph, {"nodes": nodes, "edge_prob": prob}


BUILDERS = {
    "sudoku": build_sudoku,
    "nqueens": build_nqueens,
    "latin_square": build_latin_square,
    "map_coloring": build_map,
    "futoshiki": build_futoshiki,
    "kakuro": build_kakuro,
    "custom_graph": build_custom_graph,
}


def generate_instances(per_level_override: int, seed: int, families: List[str] = None) -> Iterable[Dict[str, Any]]:
    target_families = families if families else list(PUZZLE_LEVELS.keys())
    for family, levels in PUZZLE_LEVELS.items():
        if family not in target_families:
            continue
        builder = BUILDERS[family]
        for level_cfg in levels:
            count = per_level_override if per_level_override is not None else level_cfg.get("count", 1)
            for idx in range(count):
                instance_seed = seed + hash((family, level_cfg.get("level"), idx)) % 10000
                graph, meta = builder(level_cfg, instance_seed)
                instance_id = f"{family}_{level_cfg.get('level')}_{idx}"
                yield {
                    "family": family,
                    "level": level_cfg.get("level", "unknown"),
                    "instance": instance_id,
                    "graph": graph,
                    "metadata": meta,
                }


def run_algorithms(graph: nx.Graph, algos: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for name, func in algos.items():
        start = time.perf_counter()
        try:
            coloring = func(graph)
            exec_time = time.perf_counter() - start
            valid = validate_coloring(graph, coloring)
            colors_used = count_colors(coloring) if coloring else 0
            error = None
        except Exception as exc:  # pragma: no cover - safe guard for flaky algos
            exec_time = time.perf_counter() - start
            valid = False
            colors_used = None
            coloring = {}
            error = str(exc)
        results.append({
            "algorithm": name,
            "execution_time": exec_time,
            "colors_used": colors_used,
            "valid": valid,
            "error": error,
        })
    return results


def run_suite(selected_algos: Dict[str, Any], per_level_override: int, seed: int, families: List[str] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    instances = list(generate_instances(per_level_override=per_level_override, seed=seed, families=families))

    for instance in tqdm(instances, desc="Instances", unit="graph"):
        graph = instance["graph"]
        stats = get_graph_stats(graph)
        algo_results = run_algorithms(graph, selected_algos)
        for res in algo_results:
            rows.append({
                "family": instance["family"],
                "level": instance["level"],
                "instance": instance["instance"],
                "algorithm": res["algorithm"],
                "colors_used": res["colors_used"],
                "execution_time": res["execution_time"],
                "valid": res["valid"],
                "error": res["error"],
                "num_vertices": stats.get("num_nodes"),
                "num_edges": stats.get("num_edges"),
                "density": stats.get("density"),
                "avg_degree": stats.get("avg_degree"),
                "metadata": json.dumps(instance["metadata"]),
            })
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, run_id: str, generate_plots: bool = True) -> None:
    csv_path = os.path.join(RESULTS_DIR, f"{run_id}_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    summary = df.groupby(["family", "level", "algorithm"])[["colors_used", "execution_time"]].mean().reset_index()
    summary_path = os.path.join(RESULTS_DIR, f"{run_id}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    if not generate_plots:
        return

    plt.style.use("seaborn-v0_8")
    sns.set_context("talk")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=summary, x="algorithm", y="colors_used", hue="family", ax=ax1)
    ax1.set_title("Average colors used per algorithm")
    fig1.tight_layout()
    plot1 = os.path.join(RESULTS_DIR, f"{run_id}_colors.png")
    fig1.savefig(plot1, dpi=150)
    plt.close(fig1)
    print(f"Saved plot: {plot1}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=summary, x="algorithm", y="execution_time", hue="family", ax=ax2)
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Average runtime per algorithm")
    fig2.tight_layout()
    plot2 = os.path.join(RESULTS_DIR, f"{run_id}_time.png")
    fig2.savefig(plot2, dpi=150)
    plt.close(fig2)
    print(f"Saved plot: {plot2}")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="execution_time", y="colors_used", hue="family", style="algorithm", ax=ax3, alpha=0.7)
    ax3.set_title("Efficiency: colors vs runtime")
    fig3.tight_layout()
    plot3 = os.path.join(RESULTS_DIR, f"{run_id}_efficiency.png")
    fig3.savefig(plot3, dpi=150)
    plt.close(fig3)
    print(f"Saved plot: {plot3}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full research suite across puzzles and algorithms")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--per_level", type=int, default=None, help="Override number of instances per difficulty level")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Subset of algorithms to run (default: all available)")
    parser.add_argument("--families", nargs="*", default=None, help="Subset of puzzle families to run")
    parser.add_argument("--no_plots", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algos = ALGOS
    if args.algorithms:
        selected = {name: ALGOS[name] for name in args.algorithms if name in ALGOS}
        missing = set(args.algorithms) - set(selected.keys())
        if missing:
            print(f"Warning: unknown algorithms ignored: {sorted(missing)}")
        algos = selected
    run_id = datetime.now().strftime("research_%Y%m%d_%H%M%S")
    print(f"DQN available: {DQN_AVAILABLE}")
    print(f"Algorithms: {list(algos.keys())}")
    print(f"Families: {args.families if args.families else 'all'}")
    print(f"Writing outputs to: {RESULTS_DIR}")

    df = run_suite(selected_algos=algos, per_level_override=args.per_level, seed=args.seed, families=args.families)
    print(df.head())
    save_outputs(df, run_id=run_id, generate_plots=not args.no_plots)
    print("Done.")


if __name__ == "__main__":
    main()
