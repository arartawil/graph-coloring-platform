# Algorithms Package
from .greedy import greedy_coloring
from .dsatur import dsatur_coloring
from .welsh_powell import welsh_powell_coloring
from .tabu_search import tabu_search_coloring
from .dqn import dqn_coloring

__all__ = [
    'greedy_coloring',
    'dsatur_coloring',
    'welsh_powell_coloring',
    'tabu_search_coloring',
    'dqn_coloring'
]
