"""Evaluation harness: the trained model vs a uniform-random player.

The honest, search-free test of whether the network learned to play:
a random player makes random moves, so a model that has learned anything
should beat it consistently. Win rate vs random is the headline metric.
"""

from __future__ import annotations

import random
from typing import Dict

import torch

from backend.game.board import Board, EMPTY, P1, P2


def model_move(model, board: Board, player: int) -> int:
    """Legal move from the transformer's policy head (greedy)."""
    norm = [c if c == EMPTY else (1 if c == player else 2) for c in board.cells]
    board_t = torch.tensor(norm, dtype=torch.long, device=next(model.parameters()).device)
    mask_t = torch.tensor([c == EMPTY for c in board.cells], dtype=torch.bool, device=board_t.device)
    model.eval()
    with torch.no_grad():
        _, plogits = model(board_t.unsqueeze(0), mask_t.unsqueeze(0))
    return int(plogits.squeeze(0).argmax().item())


def evaluate_vs_random(model, n: int, games: int = 40, seed: int = 0) -> Dict[str, float]:
    """Model vs a uniform-random player. Model plays first in half the games."""
    rng = random.Random(seed)
    stats = {"wins": 0, "losses": 0, "draws": 0, "illegal": 0}
    for g in range(games):
        b = Board(n)
        to_move = P1
        winner, _, over = b.outcome()
        model_first = g % 2 == 0
        while not over:
            if (to_move == P1) == model_first:
                idx = model_move(model, b, to_move)
                if b.cells[idx] != EMPTY:
                    stats["illegal"] += 1
                    winner = P2 if to_move == P1 else P1
                    break
            else:
                idx = rng.choice(b.moves())
            b.apply(idx, to_move)
            to_move = P2 if to_move == P1 else P1
            winner, _, over = b.outcome()
        if winner == EMPTY:
            stats["draws"] += 1
        elif (winner == P1) == model_first:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
    total = max(games, 1)
    return {
        "games": games,
        "wins": stats["wins"] / total,
        "losses": stats["losses"] / total,
        "draws": stats["draws"] / total,
        "illegal": stats["illegal"],
    }


def model_vs_model(model_a, model_b, n: int, games: int = 10, seed: int = 0) -> Dict[str, float]:
    """Model vs model (self-comparison), useful for RL progress."""
    rng = random.Random(seed)
    stats = {"a": 0, "b": 0, "draws": 0}
    for g in range(games):
        b = Board(n)
        to_move = P1
        winner, _, over = b.outcome()
        swap = g % 2 == 1
        while not over:
            mover = model_b if (to_move == P1) == swap else model_a
            idx = model_move(mover, b, to_move)
            if b.cells[idx] != EMPTY:
                idx = rng.choice(b.moves())
            b.apply(idx, to_move)
            to_move = P2 if to_move == P1 else P1
            winner, _, over = b.outcome()
        if winner == EMPTY:
            stats["draws"] += 1
        elif (winner == P1) == swap:
            stats["b"] += 1
        else:
            stats["a"] += 1
    total = max(games, 1)
    return {"a": stats["a"] / total, "b": stats["b"] / total, "draws": stats["draws"] / total}


def sampled_move(model, board: Board, player: int, temperature: float = 1.0) -> int:
    """Move sampled from the model's policy (used to mimic a human-like opponent)."""
    norm = [c if c == EMPTY else (1 if c == player else 2) for c in board.cells]
    x = torch.tensor(norm, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    m = torch.tensor([c == EMPTY for c in board.cells], dtype=torch.bool, device=x.device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        _, plogits = model(x, m)
    probs = torch.softmax(plogits.squeeze(0) / max(temperature, 1e-8), dim=0)
    return int(torch.multinomial(probs, 1).item())


def win_rate_against(
    model,
    opponent,
    n: int,
    games: int = 30,
    opp_temperature: float = 1.2,
    seed: int = 0,
) -> float:
    """Fraction of games `model` wins vs a sampling (human-like) opponent.

    Used to calibrate difficulty: easy/medium/hard models are tuned so their
    win-rate against the same opponent lands near 25% / 50% / 75%.
    """
    rng = random.Random(seed)
    wins = 0
    for g in range(games):
        b = Board(n)
        to_move = P1
        winner, _, over = b.outcome()
        model_first = g % 2 == 0
        while not over:
            if (to_move == P1) == model_first:
                idx = model_move(model, b, to_move)
                if b.cells[idx] != EMPTY:
                    idx = rng.choice(b.moves())
            else:
                idx = sampled_move(opponent, b, to_move, opp_temperature)
            b.apply(idx, to_move)
            to_move = P2 if to_move == P1 else P1
            winner, _, over = b.outcome()
        if winner != EMPTY and (winner == P1) == model_first:
            wins += 1
    return wins / max(1, games)


def interpolate_state(sd_strong: dict, sd_random: dict, alpha: float) -> dict:
    """Blend strong weights with random-init weights: model(alpha).

    alpha=0 -> random, alpha=1 -> strong. Strength grows monotonically,
    letting us sweep the full 0-100% win-rate range vs an opponent.
    """
    return {k: alpha * sd_strong[k] + (1 - alpha) * sd_random[k] for k in sd_strong}


def _mistake_move(model, board: Board, player: int, mistake: float, rng) -> int:
    """Greedy move with probability (1-mistake), else a random legal move."""
    if rng.random() < mistake:
        return rng.choice(board.moves())
    idx = model_move(model, board, player)
    if board.cells[idx] != EMPTY:
        return rng.choice(board.moves())
    return idx


def win_rate_against_mistake(
    model, opponent, n: int, mistake: float, games: int = 30, seed: int = 0,
    opp_mistake: float = 0.0,
) -> float:
    """Win rate of `model` (playing with `mistake` blunder rate) vs `opponent`
    playing greedily (or with its own `opp_mistake` blunder rate). Used to
    calibrate difficulty levels.
    """
    rng = random.Random(seed)
    wins = 0
    for g in range(games):
        b = Board(n)
        to_move = P1
        winner, _, over = b.outcome()
        model_first = g % 2 == 0
        while not over:
            if (to_move == P1) == model_first:
                idx = _mistake_move(model, b, to_move, mistake, rng)
            else:
                idx = _mistake_move(opponent, b, to_move, opp_mistake, rng)
            b.apply(idx, to_move)
            to_move = P2 if to_move == P1 else P1
            winner, _, over = b.outcome()
        if winner != EMPTY and (winner == P1) == model_first:
            wins += 1
    return wins / max(1, games)