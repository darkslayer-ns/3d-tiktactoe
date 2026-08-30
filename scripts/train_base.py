#!/usr/bin/env python3
"""Train a strong base model for board size N via self-play (training/selfplay.py).

Usage:
    python3 scripts/train_base.py 4        # train 4×4×4 → model_4_base.pt
    BOARD_SIZE=5 GAMES=3000 python3 scripts/train_base.py
"""

from __future__ import annotations

import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.model import ValuePolicyTransformer, value_loss
from training.selfplay import pg_self_play_game
from training.eval import evaluate_vs_random

N = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("BOARD_SIZE", 3))
GAMES = int(os.environ.get("GAMES", 4000))
TEMP = float(os.environ.get("TEMP", 0.4))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(games=GAMES, temp=TEMP, batch=128, lr=1e-3, d_model=64, layers=3, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ValuePolicyTransformer(N, d_model=d_model, nhead=8, num_layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = []
    t0 = time.time()
    best_win = -1.0
    best_state = None
    eval_every = max(1, games // 10)
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
        if g % eval_every == 0:
            r = evaluate_vs_random(model, N, games=60)
            print(f"  [{g}/{games}] vs-random win={r['wins']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            # self-play can oscillate — keep the strongest checkpoint seen
            if r["wins"] > best_win:
                best_win = r["wins"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"  best vs-random win={best_win:.2f}", flush=True)
    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    print(f"=== train base {N}x{N}x{N} on {DEVICE} ({GAMES} games) ===", flush=True)
    model = train()
    path = os.path.join(ROOT, f"model_{N}_base.pt")
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "config": {
                "n": N,
                "d_model": model.cfg()["d_model"],
                "num_layers": model.cfg()["num_layers"],
                "mistake_rate": 0.0,
                "shift_rate": 0.0,
                "wrong_move_budget": 0,
            },
        },
        path,
    )
    print(f"saved {path} ({os.path.getsize(path)//1024} KB)", flush=True)