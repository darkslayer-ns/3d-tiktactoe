#!/usr/bin/env python3
"""Fine-tune the base models on logged games (data/games.jsonl) via RL.

Each logged game is replayed: every REAL position becomes a training sample
(state from the side-to-move's perspective, the move that was actually played,
and a value target from the game outcome). The policy loss is weighted by the
mover's outcome (REINFORCE-style advantage): moves that led to wins are
reinforced, moves that led to losses are PENALIZED, draws are neutral. Only
the logged moves are used — no synthetic/augmented positions, no full
self-play retrain.

Run:  python3 scripts/finetune.py [epochs]
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.game.board import Board, EMPTY, P1, P2
from training.model import ValuePolicyTransformer, value_loss
from training.torch_loader import load_model, base_checkpoint_path
from training.eval import evaluate_vs_random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset(games):
    """(size, state, mask, move, value) for every position of every game."""
    by_size: dict[int, list] = {}
    for g in games:
        n = g["size"]
        b = Board(n)
        winner = g["winner"]
        for m in g["moves"]:
            player = m["player"]
            norm = [c if c == EMPTY else (1 if c == player else 2) for c in b.cells]
            mask = [c == EMPTY for c in b.cells]
            v = 0.5 if winner == EMPTY else (1.0 if winner == player else 0.0)
            by_size.setdefault(n, []).append((norm, mask, m["index"], v))
            b.apply(m["index"], player)
    return by_size


def finetune_size(n: int, samples, epochs: int, lr: float = 1e-4, batch: int = 256):
    """Fine-tune model_{n}_base.pt on the REAL logged positions/moves only."""
    if not samples:
        print(f"  no {n}x{n}x{n} samples, skipping", flush=True)
        return None

    base = os.path.join(ROOT, base_checkpoint_path(n))
    model, err = load_model(base, n)
    if err:
        print(f"  {base}: {err}", flush=True)
        return None
    model.train()

    states = torch.tensor([s[0] for s in samples], dtype=torch.long)
    masks = torch.tensor([s[1] for s in samples], dtype=torch.bool)
    moves = torch.tensor([s[2] for s in samples], dtype=torch.long)
    vals = torch.tensor([s[3] for s in samples], dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    B = len(states)
    print(f"  fine-tuning {n}x{n}x{n}: {B} samples from logged games", flush=True)

    # REINFORCE-style advantage from the mover's perspective:
    # +1 if that side won, -1 if it lost, 0 on draws.
    adv = 2.0 * vals - 1.0

    before = evaluate_vs_random(model, n, games=60)["wins"]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(B)
        model.train()
        tot = nb = 0
        for s in range(0, B, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            vlogits, plogits = model(states[idx].to(DEVICE), masks[idx].to(DEVICE))
            # policy loss weighted by advantage -> penalizes losing moves
            per = F.cross_entropy(plogits, moves[idx].to(DEVICE), reduction="none")
            policy_loss = (per * adv[idx].to(DEVICE)).mean()
            loss = policy_loss + value_loss(vlogits, vals[idx].to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"    epoch {ep}/{epochs} loss={tot/max(1,nb):.4f}", flush=True)

    after = evaluate_vs_random(model, n, games=60)["wins"]
    model.eval()
    torch.save(
        {
            "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
            "config": {
                "n": n,
                "d_model": model.cfg()["d_model"],
                "num_layers": model.cfg()["num_layers"],
                "mistake_rate": 0.0,
                "shift_rate": 0.0,
                "wrong_move_budget": 0,
            },
        },
        base,
    )
    print(f"  saved {base} | vs-random before={before:.2f} after={after:.2f}", flush=True)
    return base


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", nargs="?", type=int, default=2000)
    ap.add_argument("--all", action="store_true", help="include AI-vs-AI games too")
    args = ap.parse_args()

    log = os.path.join(ROOT, "data", "games.jsonl")
    if not os.path.exists(log):
        print(f"no games log at {log} — play some games first")
        sys.exit(1)
    games = [json.loads(l) for l in open(log) if l.strip()]
    if not args.all:
        keep = [g for g in games if g.get("mode") == "pve"]
        print(f"  using {len(keep)}/{len(games)} human (PvE) games "
              f"({len(games)-len(keep)} AI-vs-AI skipped; use --all to include)")
        games = keep
    print(f"== fine-tune on {len(games)} logged games ({DEVICE}) ==", flush=True)
    by_size = build_dataset(games)
    for n in sorted(by_size):
        finetune_size(n, by_size[n], args.epochs)
    print("== done ==", flush=True)


if __name__ == "__main__":
    main()