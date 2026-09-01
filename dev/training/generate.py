"""Dataset generation: random move histories for supervised training.

The supervised mode of the training lab learns purely from game history
produced by random play — no search involved. The self-play RL mode needs
no dataset at all (the model generates its own games).

States are normalized so the side-to-move is always value 1 (X).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch

from backend.game.board import Board, EMPTY, P1, P2

Data = Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
# (boards, legal_masks, move_targets, value_targets)


def _normalize(cells: List[int], to_move: int) -> List[int]:
    return [c if c == EMPTY else (1 if c == to_move else 2) for c in cells]


def _board_tensor(cells: List[int]) -> torch.Tensor:
    return torch.tensor(cells, dtype=torch.long)


def _legal_mask(cells: List[int]) -> torch.Tensor:
    return torch.tensor([c == EMPTY for c in cells], dtype=torch.bool)


def random_games(n: int, games: int, seed: int = 0) -> Data:
    """Play `games` games with uniform-random moves; keep every position.

    Value targets come from the game's true outcome. This gives the model
    volume and teaches it basic causality (finish lines, avoid losses),
    even though individual moves are random.
    """
    rng = random.Random(seed)
    boards: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    moves: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    for _ in range(games):
        b = Board(n)
        to_move = P1
        history: List[Tuple[List[int], int]] = []

        winner, _, over = b.outcome()
        while not over:
            idx = rng.choice(b.moves())
            history.append((_normalize(b.cells, to_move), idx))
            b.apply(idx, to_move)
            to_move = P2 if to_move == P1 else P1
            winner, _, over = b.outcome()

        for norm, move in history:
            occupied = sum(1 for c in norm if c != EMPTY)
            side = P1 if occupied % 2 == 0 else P2
            boards.append(_board_tensor(norm))
            masks.append(_legal_mask(norm))
            moves.append(torch.tensor(move, dtype=torch.long))
            v = 0.5 if winner == EMPTY else (1.0 if winner == side else 0.0)
            values.append(torch.tensor(v, dtype=torch.float32))

    return boards, masks, moves, values


def to_tensors(data: Data, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards, masks, moves, values = data
    return (
        torch.stack(boards).to(device),
        torch.stack(masks).to(device),
        torch.stack(moves).to(device),
        torch.stack(values).to(device),
    )