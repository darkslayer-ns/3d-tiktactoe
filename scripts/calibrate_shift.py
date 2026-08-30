#!/usr/bin/env python3
"""Calibrate the 3×3×3 difficulty levels using the 'shift by a box' mechanism.

The strong model (model_3_base.pt) picks the best box; the backend then, with
per-difficulty probability `shift_rate`, plays an adjacent empty box instead.
We sweep shift_rate against a casual human-like opponent so the in-game AI
wins about:
    easy 25% · medium 50% · hard 75% · impossible 90%
(i.e. a human wins about 75 / 50 / 25 / 10 %).

Run:  python3 scripts/calibrate_shift.py
"""

from __future__ import annotations

import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.game.board import Board, EMPTY, P1, P2
from training.eval import sampled_move

N = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def casual_human_move(model, board: Board, player: int, rng, blunder: float):
    if rng.random() < blunder:
        return rng.choice(board.moves())
    return sampled_move(model, board, player, temperature=1.0)


def ai_winrate(model, difficulty: str, shift: float, games: int, blunder: float, seed: int = 0):
    from backend.ml.model_agent import LookaheadMover
    from backend.ml.predictor import OpponentPredictor
    from training.torch_loader import TorchModelAdapter

    rng = random.Random(seed)
    engine = TorchModelAdapter(model)
    predictor = OpponentPredictor(Board(N), engine)
    model.shift_rate = shift
    wins = draws = 0
    for g in range(games):
        b = Board(N)
        ai = LookaheadMover(engine, b, predictor, difficulty=difficulty)
        player = P1
        winner, _, over = b.outcome()
        ai_first = g % 2 == 0
        while not over:
            if (player == P1) == ai_first:
                idx = ai(player)
            else:
                idx = casual_human_move(model, b, player, rng, blunder)
            b.apply(idx, player)
            player = P2 if player == P1 else P1
            winner, _, over = b.outcome()
        if winner == EMPTY:
            draws += 1
        elif (winner == P1) == ai_first:
            wins += 1
    return wins / games, draws / games


def sweep(model, difficulty, blunder, games=40, seed=0):
    out = []
    for i in range(0, 101, 10):
        s = i / 100
        r, _ = ai_winrate(model, difficulty, s, games, blunder, seed + i)
        out.append((s, r))
    return out


def calibrate(model, difficulty, target, blunder, eval_games=60):
    best = None
    for i in range(0, 101, 5):
        s = i / 100
        r, _ = ai_winrate(model, difficulty, s, eval_games, blunder, seed=3000 + i)
        if best is None or abs(r - target) < best[0]:
            best = (abs(r - target), s, r)
    return best[1]


def save_checkpoint(model, path: str, shift_rate: float) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n": N,
                "d_model": model.cfg()["d_model"],
                "num_layers": model.cfg()["num_layers"],
                "mistake_rate": 0.0,
                "shift_rate": shift_rate,
            },
        },
        path,
    )


def main():
    from training.torch_loader import load_model

    model, err = load_model(os.path.join(ROOT, "model_3_base.pt"), N)
    if err:
        print("base model error:", err)
        sys.exit(1)
    model.eval()
    blunder = float(os.environ.get("BLUNDER", 0.5))
    print(f"=== calibrate shift-rate on {DEVICE} ===", flush=True)

    print("shift-rate sweep vs casual human (AI win rate):", flush=True)
    for b in (0.4, 0.5, 0.6):
        row = sweep(model, "hard", b)
        print(f"  blunder={b}: " + "  ".join(f"s={s:.1f}->{r:.2f}" for s, r in row), flush=True)

    lo = ai_winrate(model, "hard", 1.0, 40, blunder, seed=101)[0]
    hi = ai_winrate(model, "impossible", 0.0, 40, blunder, seed=102)[0]
    print(f"  chosen blunder={blunder}: shift=1.0 -> {lo:.2f}   shift=0.0 -> {hi:.2f}", flush=True)
    if lo > 0.32:
        print("  floor too high; switching blunder 0.5 -> 0.3", flush=True)
        blunder = 0.3
    elif hi < 0.85:
        print("  ceiling too low; switching blunder 0.5 -> 0.7", flush=True)
        blunder = 0.7

    targets = {"easy": 0.25, "medium": 0.50, "hard": 0.75, "impossible": 0.90}
    print(f"  {'difficulty':<11}{'target':>8}{'shift':>8}{'ai_win':>8}{'human_win':>10}", flush=True)
    for diff, target in targets.items():
        s = 0.0 if diff == "impossible" else calibrate(model, diff, target, blunder, 60)
        ai, draw = ai_winrate(model, diff, s, games=150, blunder=blunder, seed=8888)
        path = os.path.join(ROOT, f"model_{N}_{diff}.pt")
        save_checkpoint(model, path, s)
        human = 1.0 - ai - draw
        print(f"  {diff:<11}{target:>8.2f}{s:>8.2f}{ai:>8.2f}{human:>10.2f}  (draw {draw:.2f}) -> {path}",
              flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()