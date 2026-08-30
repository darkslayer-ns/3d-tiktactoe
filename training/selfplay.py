"""Self-play training: the AI learns by playing games against itself.

No offline dataset. At every position the network samples a move from
its own policy head (temperature-controlled for exploration), and the
game's outcome becomes the value target for every stored position. The
improved network then generates better games next round.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from backend.game.board import Board, EMPTY, P1, P2


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _state(board: Board, player: int, device=None):
    norm = [c if c == EMPTY else (1 if c == player else 2) for c in board.cells]
    x = torch.tensor(norm, dtype=torch.long, device=device).unsqueeze(0)
    m = torch.tensor([c == EMPTY for c in board.cells], dtype=torch.bool, device=device).unsqueeze(0)
    return x, m


def _sample_move(model, board: Board, player: int, temperature: float) -> int:
    """Sample a legal move from the net's softmax policy."""
    x, m = _state(board, player, _device_of(model))
    with torch.no_grad():
        _, policy = model(x, m)
    logits = policy.squeeze(0)
    probs = torch.softmax(logits / temperature, dim=0)
    return int(torch.multinomial(probs, 1).item())


def self_play_game(
    model,
    n: int,
    temperature: float = 1.0,
    emit: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]], int, int]:
    """One AI-vs-AI game. Returns (samples, winner, move_count).

    Each sample: (board_tensor, legal_mask, one_hot_move, value_target).
    `emit(player, move_idx)` streams moves for the live viewer.
    """
    model.eval()
    board = Board(n)
    player = P1
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    winner, _, over = board.outcome()
    moves_played = 0

    while not over:
        move = _sample_move(model, board, player, temperature)
        x, m = _state(board, player, _device_of(model))
        target = torch.zeros(n**3, device=_device_of(model))
        target[move] = 1.0
        samples.append((x.squeeze(0), m.squeeze(0), target))
        if emit:
            emit(player, move)
        board.apply(move, player)
        player = P2 if player == P1 else P1
        moves_played += 1
        winner, _, over = board.outcome()

    # value target per position: outcome from the side-to-move's perspective
    result = []
    for x, m, target in samples:
        side = P1 if int((x != 0).sum()) % 2 == 0 else P2
        if winner == EMPTY:
            v = 0.5
        elif winner == side:
            v = 1.0
        else:
            v = 0.0
        result.append((x, m, target, v))

    return result, winner, moves_played


def pg_self_play_game(
    model,
    n: int,
    temperature: float = 0.5,
    emit: Optional[Callable[[int, int], None]] = None,
) -> Tuple[list, int]:
    """One self-play game for policy-gradient training.

    Returns (steps, winner) where each step is
    (board_tensor, legal_mask, move_idx, player_to_move). No labels yet —
    the runner turns the game outcome into advantages.
    """
    board = Board(n)
    player = P1
    steps = []
    winner, _, over = board.outcome()
    while not over:
        move = _sample_move(model, board, player, temperature)
        x, m = _state(board, player, _device_of(model))
        steps.append((x.squeeze(0), m.squeeze(0), move, player))
        if emit:
            emit(player, move)
        board.apply(move, player)
        player = P2 if player == P1 else P1
        winner, _, over = board.outcome()
    return steps, winner