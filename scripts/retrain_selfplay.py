#!/usr/bin/env python3
"""Retrain the 3×3×3 difficulty models using the existing self-play training
(training/selfplay.py), then calibrate difficulty against the real game
mover so a human wins about:

    easy 75% · medium 50% · hard 25% · impossible 10%

Calibration measures the ACTUAL in-game AI (LookaheadMover + mistake rate)
against a casual human-like opponent, unlike the old runner which measured
the raw policy and produced a lopsided ladder.

Run:  python3 scripts/retrain_selfplay.py
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
from training.model import ValuePolicyTransformer, value_loss
from training.selfplay import pg_self_play_game
from training.eval import evaluate_vs_random, sampled_move

N = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# 1. self-play training (REINFORCE, from training/selfplay.py)
# --------------------------------------------------------------------------
def train_selfplay(games=2000, temp=0.5, batch=128, lr=1e-3, d_model=64, layers=3, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ValuePolicyTransformer(N, d_model=d_model, nhead=8, num_layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = []
    t0 = time.time()
    for g in range(1, games + 1):
        steps, winner = pg_self_play_game(model, N, temperature=temp)
        for x, m, idx, player in steps:
            adv = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            buffer.append((x.to(DEVICE), m.to(DEVICE), idx, adv,
                           1.0 if adv > 0 else (0.5 if adv == 0 else 0.0)))
        if len(buffer) >= batch:
            sel = random.sample(buffer, batch)
            xb = torch.stack([s[0] for s in sel])
            mb = torch.stack([s[1] for s in sel])
            mv = torch.tensor([s[2] for s in sel], device=DEVICE)
            adv = torch.tensor([s[3] for s in sel], dtype=torch.float32, device=DEVICE)
            vt = torch.tensor([s[4] for s in sel], dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            vl, pl = model(xb, mb)
            logp = torch.log_softmax(pl, dim=1).gather(1, mv.unsqueeze(1)).squeeze(1)
            loss = -(logp * adv).mean() + value_loss(vl, vt)
            loss.backward()
            opt.step()
            del buffer[:batch]
        if g % max(1, games // 5) == 0:
            r = evaluate_vs_random(model, N, games=40)
            print(f"  [{g}/{games}] vs-random win={r['wins']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return model


# --------------------------------------------------------------------------
# 2. difficulty calibration vs the REAL game mover
# --------------------------------------------------------------------------
def casual_human_move(model, board: Board, player: int, rng, blunder: float):
    if rng.random() < blunder:
        return rng.choice(board.moves())
    return sampled_move(model, board, player, temperature=1.0)


def ai_winrate(model, difficulty: str, mistake: float, games: int, blunder: float, seed: int = 0):
    from backend.ml.model_agent import LookaheadMover
    from backend.ml.predictor import OpponentPredictor
    from training.torch_loader import TorchModelAdapter

    rng = random.Random(seed)
    engine = TorchModelAdapter(model)
    predictor = OpponentPredictor(Board(N), engine)
    model.mistake_rate = mistake
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


def calibrate(model, difficulty, target, blunder, eval_games=60):
    best = None
    for i in range(0, 101, 5):
        m = i / 100
        r, _ = ai_winrate(model, difficulty, m, eval_games, blunder, seed=2000 + i)
        if best is None or abs(r - target) < best[0]:
            best = (abs(r - target), m, r)
    return best[1]


def save_checkpoint(model, path: str, mistake_rate: float) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n": N,
                "d_model": model.cfg()["d_model"],
                "num_layers": model.cfg()["num_layers"],
                "mistake_rate": mistake_rate,
            },
        },
        path,
    )


def main():
    games = int(os.environ.get("GAMES", 2000))
    blunder = float(os.environ.get("BLUNDER", 0.5))
    print(f"=== retrain 3x3x3 on {DEVICE} (self-play {games} games) ===", flush=True)

    print("[1/3] self-play training", flush=True)
    model = train_selfplay(games=games)
    save_checkpoint(model, os.path.join(ROOT, "model_3_base.pt"), 0.0)

    print("[2/3] probing difficulty range vs casual human", flush=True)
    lo, _ = ai_winrate(model, "hard", 1.0, 40, blunder, seed=101)
    hi, _ = ai_winrate(model, "impossible", 0.0, 40, blunder, seed=102)
    print(f"  blunder={blunder}: m=1.0 ai_win={lo:.2f}   m=0.0 ai_win={hi:.2f}", flush=True)
    if lo > 0.32:
        print(f"  floor {lo:.2f} too high for easy=25%; using blunder=0.3", flush=True)
        blunder = 0.3
    elif hi < 0.85:
        print(f"  ceiling {hi:.2f} too low for impossible=90%; using blunder=0.7", flush=True)
        blunder = 0.7
    print(f"  using casual-human blunder={blunder}", flush=True)

    print("[3/3] calibrating (target = AI win rate vs casual human)", flush=True)
    targets = {"easy": 0.25, "medium": 0.50, "hard": 0.75, "impossible": 0.90}
    print(f"  {'difficulty':<11}{'target':>8}{'mistake':>9}{'ai_win':>8}{'human_win':>10}", flush=True)
    for diff, target in targets.items():
        m = 0.0 if diff == "impossible" else calibrate(model, diff, target, blunder, 60)
        ai, draw = ai_winrate(model, diff, m, games=150, blunder=blunder, seed=7777)
        path = os.path.join(ROOT, f"model_{N}_{diff}.pt")
        save_checkpoint(model, path, m)
        human = 1.0 - ai - draw
        print(f"  {diff:<11}{target:>8.2f}{m:>9.2f}{ai:>8.2f}{human:>10.2f}  (draw {draw:.2f}) -> {path}",
              flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()