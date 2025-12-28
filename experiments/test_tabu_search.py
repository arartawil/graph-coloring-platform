"""Experiment suite for adaptive Tabu Search graph coloring.

Runs Tabu Search across graph families, tunes parameters, compares against
classical algorithms (Greedy, DSatur, Welsh-Powell), and exports results.
"""

from __future__ import annotations

import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.stats import ttest_ind

from algorithms import (
    greedy_coloring,
    dsatur_coloring,
    welsh_powell_coloring,
    TabuSearchColoring,
)

OUTPUT_DIR = Path("experiments/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def graph_families(n: int = 40) -> Dict[str, nx.Graph]:
    """Generate a suite of graph types for testing."""
    graphs: Dict[str, nx.Graph] = {}
    graphs["dense_gnp"] = nx.gnp_random_graph(n, 0.2, seed=42)
    graphs["sparse_gnp"] = nx.gnp_random_graph(n, 0.04, seed=21)
    graphs["regular"] = nx.random_regular_graph(4, n, seed=7)
    graphs["erdos_medium"] = nx.gnp_random_graph(n, 0.1, seed=99)
    return graphs


def run_algorithm(name: str, graph: nx.Graph) -> Dict[str, Any]:
    start = time.perf_counter()
    if name == "greedy":
        coloring = greedy_coloring(graph)
    elif name == "dsatur":
        coloring = dsatur_coloring(graph)
    elif name == "welsh_powell":
        coloring = welsh_powell_coloring(graph)
    elif name == "tabu":
        algo = TabuSearchColoring(graph, max_iterations=500, tabu_tenure=50)
        coloring = algo.color()
    else:
        raise ValueError(f"Unknown algorithm {name}")
    exec_time = time.perf_counter() - start
    conflicts = sum(1 for u, v in graph.edges() if coloring.get(u) == coloring.get(v))
    colors_used = len(set(coloring.values())) if coloring else 0
    return {
        "algorithm": name,
        "colors_used": colors_used,
        "execution_time": exec_time,
        "conflicts": conflicts,
        "coloring": coloring,
    }


def parameter_grid() -> List[Tuple[int, int]]:
    tenures = [10, 25, 50, 100]
    max_iters = [100, 500, 1000, 5000]
    return list(itertools.product(tenures, max_iters))


def tune_tabu(graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for g_name, g in graphs.items():
        for tenure, max_iter in parameter_grid():
            algo = TabuSearchColoring(g, max_iterations=max_iter, tabu_tenure=tenure)
            result = algo.color()
            conflicts = algo.best_conflicts
            colors_used = len(set(result.values())) if result else 0
            rows.append({
                "graph": g_name,
                "tabu_tenure": tenure,
                "max_iterations": max_iter,
                "best_conflicts": conflicts,
                "colors_used": colors_used,
                "execution_time": algo.execution_time,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "tabu_tuning.csv", index=False)
    return df


def compare_algorithms(graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    algos = ["greedy", "dsatur", "welsh_powell", "tabu"]
    for g_name, g in graphs.items():
        for algo in algos:
            res = run_algorithm(algo, g)
            rows.append({
                "graph": g_name,
                **{k: v for k, v in res.items() if k != "coloring"},
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)
    return df


def statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    greedy = df[df.algorithm == "greedy"]["colors_used"].tolist()
    tabu = df[df.algorithm == "tabu"]["colors_used"].tolist()
    if len(greedy) > 1 and len(tabu) > 1:
        stat, p = ttest_ind(greedy, tabu, equal_var=False)
        results["greedy_vs_tabu_colors"] = {"stat": stat, "p_value": p}
    return results


def plot_convergence_example(graph: nx.Graph) -> None:
    algo = TabuSearchColoring(graph, max_iterations=1000, tabu_tenure=50)
    algo.color()
    fig = algo.plot_convergence()
    if fig:
        fig.savefig(OUTPUT_DIR / "tabu_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_summary_json(df_compare: pd.DataFrame, stats: Dict[str, Any]) -> None:
    summary = {
        "best_by_graph": df_compare.sort_values("colors_used").groupby("graph").first().to_dict(),
        "stats": stats,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    graphs = graph_families()
    tuning_df = tune_tabu(graphs)
    comparison_df = compare_algorithms(graphs)
    stats = statistical_tests(comparison_df)
    plot_convergence_example(graphs["dense_gnp"])
    save_summary_json(comparison_df, stats)
    print("Tuning saved to", OUTPUT_DIR / "tabu_tuning.csv")
    print("Comparison saved to", OUTPUT_DIR / "algorithm_comparison.csv")
    print("Summary saved to", OUTPUT_DIR / "summary.json")
    print("Convergence plot saved to", OUTPUT_DIR / "tabu_convergence.png")


+if __name__ == "__main__":
+    main()
