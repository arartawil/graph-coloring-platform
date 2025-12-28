# Algorithms Package
from .greedy import greedy_coloring, greedy_with_order, calculate_metrics, GreedyColoringAlgorithm
from .dsatur import dsatur_coloring, calculate_saturation, DSaturAlgorithm
from .welsh_powell import welsh_powell_coloring, sort_by_degree, find_independent_set, WelshPowellAlgorithm
from .smallest_last import smallest_last_coloring, SmallestLastAlgorithm
from .largest_first import largest_first_coloring, LargestFirstAlgorithm
from .iterated_greedy import iterated_greedy_coloring, IteratedGreedyAlgorithm
from .tabu_search import tabu_search_coloring, TabuSearchColoring

try:
    from .dqn import dqn_coloring
    DQN_AVAILABLE = True
except Exception:
    DQN_AVAILABLE = False

    def dqn_coloring(*args, **kwargs):
        """Placeholder when DQN dependencies are missing."""
        raise ImportError("DQN coloring is unavailable. Install torch and related deps to enable.")

__all__ = [
    'greedy_coloring',
    'greedy_with_order',
    'calculate_metrics',
    'GreedyColoringAlgorithm',
    'dsatur_coloring',
    'calculate_saturation',
    'DSaturAlgorithm',
    'welsh_powell_coloring',
    'sort_by_degree',
    'find_independent_set',
    'WelshPowellAlgorithm',
    'smallest_last_coloring',
    'SmallestLastAlgorithm',
    'largest_first_coloring',
    'LargestFirstAlgorithm',
    'iterated_greedy_coloring',
    'IteratedGreedyAlgorithm',
    'tabu_search_coloring',
    'TabuSearchColoring',
    'dqn_coloring',
    'DQN_AVAILABLE'
]
