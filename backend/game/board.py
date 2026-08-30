"""n×n×n 3D tic-tac-toe board core.

Board indexed as (x, y, z) with x,y,z in [0, n). Cells are stored
in a flat array at index x + n*(y + n*z).

Values: 0 empty, 1 player 1 (X), 2 player 2 (O).
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, List, Optional, Sequence, Tuple

Cell = int
Coord = Tuple[int, int, int]

EMPTY = 0
P1 = 1
P2 = 2

SUPPORTED_SIZES = (3, 4, 5, 6)


def line_count(n: int) -> int:
    """Number of winning lines for an n×n×n board: ((n+2)^3 - n^3)/2."""
    return ((n + 2) ** 3 - n**3) // 2


def _directions(n: int) -> List[Tuple[int, int, int]]:
    """All primitive direction vectors (no opposites, zero excluded)."""
    dirs = []
    for d in product((-1, 0, 1), repeat=3):
        if d == (0, 0, 0):
            continue
        if d[0] > 0 or (d[0] == 0 and d[1] > 0) or (d[0] == 0 and d[1] == 0 and d[2] > 0):
            dirs.append(d)
    return dirs


_LINES_CACHE: dict[int, List[Tuple[Coord, ...]]] = {}


def build_lines(n: int) -> List[Tuple[Coord, ...]]:
    """All winning lines: n consecutive cells along a direction vector.

    Results are cached per board size.
    """
    if n in _LINES_CACHE:
        return _LINES_CACHE[n]
    dirs = _directions(n)
    lines: List[Tuple[Coord, ...]] = []
    for start in product(range(n), repeat=3):
        for dx, dy, dz in dirs:
            pts = [(start[0] + dx * k, start[1] + dy * k, start[2] + dz * k) for k in range(n)]
            if all(0 <= c < n for p in pts for c in p):
                lines.append(tuple(pts))
    _LINES_CACHE[n] = lines
    return lines


class Board:
    __slots__ = ("n", "cells", "lines")

    def __init__(self, n: int, cells: Optional[Sequence[int]] = None):
        if n not in SUPPORTED_SIZES:
            raise ValueError(f"unsupported board size {n} (supported: {list(SUPPORTED_SIZES)})")
        self.n = n
        self.cells = list(cells) if cells is not None else [EMPTY] * (n**3)
        self.lines = build_lines(n)

    # -- indexing ----------------------------------------------------------
    def idx(self, x: int, y: int, z: int) -> int:
        return x + self.n * (y + self.n * z)

    def get(self, x: int, y: int, z: int) -> Cell:
        return self.cells[self.idx(x, y, z)]

    def set(self, x: int, y: int, z: int, value: Cell) -> None:
        self.cells[self.idx(x, y, z)] = value

    def coord(self, i: int) -> Coord:
        n = self.n
        z, r = divmod(i, n * n)
        y, x = divmod(r, n)
        return (x, y, z)

    def coord_of_line(self, line: Sequence[Coord]) -> List[int]:
        return [self.idx(*c) for c in line]

    # -- state -------------------------------------------------------------
    def is_empty(self, x: int, y: int, z: int) -> bool:
        return self.get(x, y, z) == EMPTY

    def empty_count(self) -> int:
        return self.cells.count(EMPTY)

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def moves(self) -> List[int]:
        """Indices of empty cells."""
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def move_count(self) -> int:
        return self.cells.count(EMPTY)

    def copy(self) -> "Board":
        return Board(self.n, self.cells)

    def apply(self, index: int, player: Cell) -> None:
        if self.cells[index] != EMPTY:
            raise ValueError(f"cell {index} already occupied")
        self.cells[index] = player

    # -- outcome -----------------------------------------------------------
    def winner(self) -> Tuple[Cell, Optional[List[Coord]]]:
        """Return (player, winning line coords) or (EMPTY, None)."""
        for line in self.lines:
            vals = [self.get(*c) for c in line]
            if vals[0] != EMPTY and all(v == vals[0] for v in vals):
                return (vals[0], list(line))
        return (EMPTY, None)

    def outcome(self) -> Tuple[Cell, Optional[List[Coord]], bool]:
        """Return (winner, winning line, game_over)."""
        player, line = self.winner()
        if player != EMPTY:
            return (player, line, True)
        if self.is_full():
            return (EMPTY, None, True)
        return (EMPTY, None, False)

    def signature(self) -> Tuple[int, Tuple[int, ...]]:
        return (self.n, tuple(self.cells))