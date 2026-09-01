#!/usr/bin/env python3
"""Calibrate the 3×3×3 difficulty models against the real game mover.

The strong model (model_3_base.pt) is loaded; the backend deliberately plays
a random move with per-difficulty probability `mistake_rate` (this now
overrides even immediate wins/blocks, so high rates are genuinely weak).
We sweep mistake_rate vs a casual human-like opponent so the in-game AI wins
about:
    easy 25% · medium 50% · hard 75% · impossible 90%
i.e. a human wins about 75 / 50 / 25 / 10 %.

Run:  python3 scripts/calibrate_models.py
"""

from __future__ import annotations

import os
import random
import sys

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


def ai_winrate(model, difficulty, mistake, games, blunder, seed=0):
    from backend.ml.model_agent import LookaheadMover
    from backend.ml.predictor import OpponentPredictor
    from training.torch_loader import TorchModelAdapter

    rng = random.Random(seed)
    engine = TorchModelAdapter(model)
    predictor = OpponentPredictor(Board(N), engine)
    model.mistake_rate = mistake
    model.shift_rate = 0.0
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


def calibrate(model, difficulty, target, blunder, eval_games=40):
    best = None
    for i in range(0, 101, 5):
        m = i / 100
        r, _ = ai_winrate(model, difficulty, m, eval_games, blunder, seed=4000 + i)
        if best is None or abs(r - target) < best[0]:
            best = (abs(r - target), m, r)
    return best[1]


def save_checkpoint(model, path, mistake_rate) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n": N,
                "d_model": model.cfg()["d_model"],
                "num_layers": model.cfg()["num_layers"],
                "mistake_rate": mistake_rate,
                "shift_rate": 0.0,
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
    blunder = float(os.environ.get("BLUNDER", 0.25))
    print(f"=== calibrate 3x3x3 difficulty on {DEVICE} (casual-human blunder={blunder}) ===", flush=True)

    lo, _ = ai_winrate(model, "hard", 1.0, 60, blunder, seed=101)
    hi, _ = ai_winrate(model, "impossible", 0.0, 40, blunder, seed=102)
    print(f"  floor (pure random)={lo:.2f}   ceiling (strong)={hi:.2f}", flush=True)
    if lo > 0.30:
        print("  floor too high for easy=25%; switching blunder 0.25 -> 0.2", flush=True)
        blunder = 0.2
    elif hi < 0.88:
        print("  ceiling too low for impossible=90%; switching blunder 0.25 -> 0.35", flush=True)
        blunder = 0.35

    targets = {"easy": 0.25, "medium": 0.50, "hard": 0.75, "impossible": 0.90}
    print(f"  {'difficulty':<11}{'target':>8}{'mistake':>9}{'ai_win':>8}{'human_win':>10}", flush=True)
    for diff, target in targets.items():
        m = 0.0 if diff == "impossible" else calibrate(model, diff, target, blunder, 40)
        ai, draw = ai_winrate(model, diff, m, games=150, blunder=blunder, seed=9999)
        path = os.path.join(ROOT, f"model_{N}_{diff}.pt")
        save_checkpoint(model, path, m)
        human = 1.0 - ai - draw
        print(f"  {diff:<11}{target:>8.2f}{m:>9.2f}{ai:>8.2f}{human:>10.2f}  (draw {draw:.2f}) -> {path}",
              flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()