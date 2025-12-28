# Puzzles Package
from .sudoku import SudokuPuzzle
from .nqueens import NQueensPuzzle
from .kakuro import KakuroPuzzle
from .futoshiki import FutoshikiPuzzle
from .latin_square import LatinSquarePuzzle
from .map_coloring import MapColoringPuzzle
from .custom_graph import CustomGraphPuzzle

__all__ = [
    'SudokuPuzzle',
    'NQueensPuzzle',
    'KakuroPuzzle',
    'FutoshikiPuzzle',
    'LatinSquarePuzzle',
    'MapColoringPuzzle',
    'CustomGraphPuzzle'
]
