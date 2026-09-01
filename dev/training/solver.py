"""Alpha-beta search solver for n×n×n tic-tac-toe.

The "teacher" for search distillation: for any board it returns the
game-theoretically best move (3x3x3 solves perfectly; larger sizes use
iterative deepening with a time/depth budget and a line heuristic).

Values are from the side-to-move's perspective:
  +WIN  = side to move wins with perfect play
  -WIN  = side to move loses (opponent wins)
  0     = draw
  small heuristic values when depth-limited
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from backend.game.board import Board, EMPTY, P1, P2

WIN = 1_000_000


class Solver:
    def __init__(self, n: int):
        self.n = n
        self.table: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.nodes = 0

    def reset(self) -> None:
        self.table.clear()
        self.nodes = 0

    def best_move(self, board: Board, player: int, max_depth: int | None = None) -> Tuple[int, int]:
        if max_depth is None:
            max_depth = self.n * self.n  # enough plies to see any forced line
        """Return (move, value-from-player-perspective)."""
        opp = P2 if player == P1 else P1
        # immediate win
        for m in board.moves():
            board.apply(m, player)
            w, _, over = board.outcome()
            board.cells[m] = EMPTY
            if w == player:
                return m, WIN
        # immediate block
        for m in board.moves():
            board.apply(m, opp)
            w, _, over = board.outcome()
            board.cells[m] = EMPTY
            if w == opp:
                return m, -WIN

        best_m = board.moves()[0]
        best_v = -WIN - 1
        moves = self._ordered(board, player)
        for m in moves:
            board.apply(m, player)
            v = -self._search(board, opp, max_depth - 1, -WIN - 1, WIN + 1)
            board.cells[m] = EMPTY
            if v > best_v:
                best_v, best_m = v, m
                if v >= WIN:
                    break
        return best_m, best_v

    def _search(self, board: Board, player: int, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        winner, _, over = board.outcome()
        if over:
            if winner == EMPTY:
                return 0
            return WIN if winner == player else -WIN
        if depth <= 0:
            return self._heuristic(board, player)

        key = (tuple(board.cells), player)
        cached = self.table.get(key)
        if cached is not None:
            return cached

        opp = P2 if player == P1 else P1
        best = -WIN - 1
        for m in self._ordered(board, player):
            board.apply(m, player)
            v = -self._search(board, opp, depth - 1, -beta, -alpha)
            board.cells[m] = EMPTY
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        self.table[key] = best
        return best

    def _ordered(self, board: Board, player: int) -> list:
        """Move ordering: center-ish first, then by line potential."""
        center = (self.n - 1) / 2
        scored = []
        for idx in board.moves():
            x, y, z = board.coord(idx)
            d = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
            scored.append((d, idx))
        scored.sort()
        return [i for _, i in scored]

    def _heuristic(self, board: Board, player: int) -> int:
        """Line-based heuristic at depth limit (cheap, signed)."""
        opp = P2 if player == P1 else P1
        score = 0
        for line in board.lines:
            own = oppn = 0
            for c in line:
                v = board.get(*c)
                if v == player:
                    own += 1
                elif v == opp:
                    oppn += 1
            if own and oppn:
                continue
            if own:
                score += 10 ** (own - 1) if own > 1 else 1
            elif oppn:
                score -= 10 ** (oppn - 1) if oppn > 1 else 1
        return score


def solve_outcome(board: Board, player: int, max_depth: int | None = None) -> int:
    """True game value for the side to move (WIN / -WIN / 0)."""
    s = Solver(board.n)
    _, v = s.best_move(board, player, max_depth)
    return v