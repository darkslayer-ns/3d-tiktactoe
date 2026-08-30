#!/usr/bin/env python3
"""Train + calibrate the 3×3×3 difficulty models.

Pipeline:
  1. scripts/gentrain (Go) generates positions with an alpha-beta teacher
     into a compact binary file (32 bytes/record).
  2. This script reads them, augments 48× (cube symmetries), trains the
     policy/value transformer on CUDA, then calibrates each difficulty's
     "mistake rate" against the real game mover (LookaheadMover) so the AI
     wins roughly 25 / 50 / 75 / 90 % vs a casual human-like opponent
     (human win rates ≈ 75 / 50 / 25 / 10 %).
  3. Saves model_3_easy.pt, model_3_medium.pt, model_3_hard.pt,
     model_3_impossible.pt (+ model_3_base.pt for re-calibration).

Run:  python3 scripts/retrain.py <train_data.bin>
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.game.board import Board, EMPTY, P1, P2
from training.model import ValuePolicyTransformer, value_loss
from training.eval import evaluate_vs_random, sampled_move

N = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# symmetry maps (48 cube symmetries: Q[j]=source cell for output j,
# F[m]=output cell for a move played at source m)
# --------------------------------------------------------------------------
def symmetry_maps(n: int):
    idx = {}
    for x in range(n):
        for y in range(n):
            for z in range(n):
                idx[(x, y, z)] = x + n * (y + n * z)
    out = []
    seen = set()
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product([1, -1], repeat=3):
            fwd = []
            ok = True
            for i in range(n**3):
                z, r = divmod(i, n * n)
                y, x = divmod(r, n)
                c = (x, y, z)
                t = []
                for k in range(3):
                    v = c[perm[k]]
                    if flips[k] == -1:
                        v = n - 1 - v
                    t.append(v)
                if any(v < 0 or v >= n for v in t):
                    ok = False
                    break
                fwd.append(idx[tuple(t)])
            if ok:
                key = tuple(fwd)
                if key in seen:
                    continue
                seen.add(key)
                q = [0] * (n**3)
                for i, j in enumerate(fwd):
                    q[j] = i
                out.append((q, fwd))
    return out


SYMS = symmetry_maps(N)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_data(path: str):
    d = np.fromfile(path, dtype=np.uint8)
    n = len(d) // 32
    if n == 0:
        raise SystemExit(f"no records in {path}")
    a = d[: n * 32].reshape(n, 32)
    states = a[:, :27].astype(np.int64)
    moves = a[:, 27].astype(np.int64)
    vals = np.frombuffer(a[:, 28:32].tobytes(), dtype="<f4").astype(np.float32)
    print(f"  loaded {n} records from {path}", flush=True)
    return states, moves, vals


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(states, moves, vals, epochs=8, d_model=64, layers=3, batch=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    model = ValuePolicyTransformer(N, d_model=d_model, nhead=8, num_layers=layers).to(DEVICE)
    print(f"  device={DEVICE} params={sum(p.numel() for p in model.parameters()):,} "
          f"augment={len(SYMS)}x samples={len(states)*len(SYMS):,}", flush=True)

    xs = torch.from_numpy(states).to(DEVICE)          # (B, 27)
    masks = torch.from_numpy(states != 0).to(DEVICE)  # (B, 27)
    mv = torch.from_numpy(moves).to(DEVICE)
    vs = torch.from_numpy(vals).to(DEVICE)

    # materialize the 48x-augmented dataset once (GPU)
    qs = [torch.tensor(Q, dtype=torch.long, device=DEVICE) for Q, _ in SYMS]
    fwd = [torch.tensor(F, dtype=torch.long, device=DEVICE) for F, _ in SYMS]
    x_parts, m_parts, mv_parts, v_parts = [], [], [], []
    for Q, Fm in zip(qs, fwd):
        x_parts.append(xs.index_select(1, Q))
        m_parts.append(masks.index_select(1, Q))
        mv_parts.append(Fm.index_select(0, mv))
        v_parts.append(vs)
    xs = torch.cat(x_parts, 0)
    masks = torch.cat(m_parts, 0)
    mv = torch.cat(mv_parts, 0)
    vs = torch.cat(v_parts, 0)
    B = len(xs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(B)
        total_loss = nbatch = 0
        model.train()
        t0 = time.time()
        for s in range(0, B, batch):
            idx = perm[s : s + batch]
            xb = xs[idx]
            mb = masks[idx]
            mvb = mv[idx]
            vb = vs[idx]
            opt.zero_grad()
            vlogits, plogits = model(xb, mb)
            loss = F.cross_entropy(plogits, mvb) + value_loss(vlogits, vb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nbatch += 1
        r = evaluate_vs_random(model, N, games=60)
        print(
            f"  epoch {ep}/{epochs} loss={total_loss/max(1,nbatch):.4f} "
            f"vs-random win={r['wins']:.2f} loss={r['losses']:.2f} draw={r['draws']:.2f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
    return model


# --------------------------------------------------------------------------
# difficulty calibration vs the REAL game mover
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


def calibrate(model, difficulty: str, target: float, blunder: float, eval_games: int = 60):
    best = None
    for i in range(0, 101, 5):
        m = i / 100
        r, _ = ai_winrate(model, difficulty, m, eval_games, blunder, seed=2000 + i)
        err = abs(r - target)
        if best is None or err < best[0]:
            best = (err, m, r)
    return best[1], best[2]


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
    if len(sys.argv) < 2:
        print("usage: python3 scripts/retrain.py <train_data.bin>")
        sys.exit(1)
    data_path = sys.argv[1]
    epochs = int(os.environ.get("EPOCHS", 8))
    blunder = float(os.environ.get("BLUNDER", 0.5))
    print(f"=== retrain 3x3x3 on {DEVICE} ===", flush=True)

    print("[1/3] loading data", flush=True)
    states, moves, vals = load_data(data_path)

    print("[2/3] training", flush=True)
    model = train(states, moves, vals, epochs=epochs)
    save_checkpoint(model, os.path.join(ROOT, "model_3_base.pt"), 0.0)

    print("[3/3] calibrating difficulties (target = AI win rate vs casual human)", flush=True)
    lo, hi = ai_winrate(model, "hard", 1.0, 40, blunder, seed=101)[0], \
             ai_winrate(model, "impossible", 0.0, 40, blunder, seed=102)[0]
    print(f"  probe blunder={blunder}: m=1.0 ai_win={lo:.2f}   m=0.0 ai_win={hi:.2f}", flush=True)
    if lo > 0.32:
        print(f"  floor {lo:.2f} too high for easy=25%; switching to blunder=0.3", flush=True)
        blunder = 0.3
    elif hi < 0.85:
        print(f"  ceiling {hi:.2f} too low for impossible=90%; switching to blunder=0.7", flush=True)
        blunder = 0.7
    print(f"  using casual-human blunder={blunder}", flush=True)

    targets = {"easy": 0.25, "medium": 0.50, "hard": 0.75, "impossible": 0.90}
    print(f"  {'difficulty':<11}{'target':>8}{'mistake':>9}{'ai_win':>8}{'human_win':>10}", flush=True)
    for diff, target in targets.items():
        m = 0.0 if diff == "impossible" else calibrate(model, diff, target, blunder, 60)[0]
        ai, draw = ai_winrate(model, diff, m, games=150, blunder=blunder, seed=7777)
        path = os.path.join(ROOT, f"model_{N}_{diff}.pt")
        save_checkpoint(model, path, m)
        human = 1.0 - ai - draw
        print(f"  {diff:<11}{target:>8.2f}{m:>9.2f}{ai:>8.2f}{human:>10.2f}  (draw {draw:.2f}) -> {path}", flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()