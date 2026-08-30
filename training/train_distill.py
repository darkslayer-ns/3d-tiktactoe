"""Search-distillation trainer.

Trains the size-agnostic transformer to imitate the Go alpha-beta solver
(backend/distill). Each sample is a board position labelled with the
solver's best move + game value.

Train/eval are split by WHOLE GAMES (game_id in the binary), so eval
positions never come from training games — no leakage.

Run:
  go run ./distill -n 3 -games 20000 -out /tmp/d3.bin
  python3 -m training.train_distill /tmp/d3.bin
"""

from __future__ import annotations

import argparse
import random
import struct
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.model import ValuePolicyTransformer, policy_loss, value_loss
from training.eval import evaluate_vs_random


def load_go_data(path: str):
    """Parse the Go distill binary. Returns (games, samples).

    sample: {cells, move, value, game}
    """
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<i", data, 0)[0]
    n_tokens = n**3
    rec = n_tokens + 4 + 4 + 4  # cells + move + value + game_id
    off = 12
    samples = []
    games = 0
    while off + rec <= len(data):
        cells = list(struct.unpack_from(f"<{n_tokens}b", data, off))
        move = struct.unpack_from("<i", data, off + n_tokens)[0]
        value = struct.unpack_from("<i", data, off + n_tokens + 4)[0]
        game = struct.unpack_from("<i", data, off + n_tokens + 8)[0]
        games = max(games, game)
        samples.append({"cells": cells, "move": move, "value": value, "game": game})
        off += rec
    return n, samples, games


def normalize(cells, to_move):
    """Rewrite so to_move is X (1) and opponent is O (2)."""
    return [c if c == 0 else (1 if c == to_move else 2) for c in cells]


def value_target(v: int) -> float:
    """Solver value -> side-to-move win probability target."""
    if v >= 1_000_000:
        return 1.0
    if v <= -1_000_000:
        return 0.0
    return 0.5


def build_tensors(samples, n, device, tm_fn, draw_weight: float = 1.0):
    boards, masks, moves, values = [], [], [], []
    for s in samples:
        is_draw = abs(s["value"]) < 1000
        reps = int(draw_weight) if is_draw else 1
        tm = tm_fn(s["cells"])
        norm = normalize(s["cells"], tm)
        bt = torch.tensor(norm, dtype=torch.long)
        bm = torch.tensor([c == 0 for c in norm], dtype=torch.bool)
        bmv = torch.tensor(s["move"], dtype=torch.long)
        bv = torch.tensor(value_target(s["value"]), dtype=torch.float32)
        for _ in range(reps):
            boards.append(bt)
            masks.append(bm)
            moves.append(bmv)
            values.append(bv)
    return (
        torch.stack(boards).to(device),
        torch.stack(masks).to(device),
        torch.stack(moves).to(device),
        torch.stack(values).to(device),
    )


def to_move_of(cells):
    occupied = sum(1 for c in cells if c != 0)
    return 1 if occupied % 2 == 0 else 2


def log_progress(msg: str, path: str | None) -> None:
    print(msg, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(msg + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="path to Go distill binary")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--save", type=str, default="model_universal.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--progress", type=str, default="/tmp/opencode/train_distill.log",
                   help="append live progress here for non-blocking monitoring")
    p.add_argument("--resume", action="store_true", help="load existing --save and continue from saved epoch")
    p.add_argument("--max-epochs", type=int, default=200, help="absolute epoch cap for resume")
    p.add_argument("--draw-weight", type=int, default=200,
                   help="how many times to duplicate each draw sample (fix rare draws)")
    args = p.parse_args()

    n, samples, games = load_go_data(args.data)
    log_progress(f"n={n} games={games} samples={len(samples)} device={args.device}", args.progress)
    device = args.device

    # split by WHOLE GAME
    game_ids = list(range(1, games + 1))
    random.Random(args.seed).shuffle(game_ids)
    n_eval_games = max(1, int(len(game_ids) * args.eval_frac))
    eval_games = set(game_ids[:n_eval_games])
    train_samples = [s for s in samples if s["game"] not in eval_games]
    eval_samples = [s for s in samples if s["game"] in eval_games]
    log_progress(f"train samples: {len(train_samples)}  eval samples: {len(eval_samples)}  (split by game)", args.progress)

    tb, tm, tmv, tv = build_tensors(train_samples, n, device, to_move_of, draw_weight=args.draw_weight)
    eb, em, emv, ev = build_tensors(eval_samples, n, device, to_move_of)
    train_ds = DataLoader(TensorDataset(tb, tm, tmv, tv), batch_size=args.batch, shuffle=True)
    eval_ds = DataLoader(TensorDataset(eb, em, emv, ev), batch_size=args.batch)

    model = ValuePolicyTransformer(n=n, d_model=args.d_model, num_layers=args.layers).to(device)
    start_epoch = 1
    if args.resume:
        import os

        if os.path.exists(args.save):
            ck = torch.load(args.save, map_location=device)
            model.load_state_dict(ck["state_dict"])
            start_epoch = ck.get("config", {}).get("epoch", 0) + 1
            log_progress(f"resumed from epoch {start_epoch - 1}", args.progress)
        else:
            log_progress("--resume but no checkpoint; starting fresh", args.progress)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_progress(f"params: {sum(p.numel() for p in model.parameters()):,} on {device}", args.progress)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0 = time.time()
        for b, m, mv, v in train_ds:
            opt.zero_grad()
            vl, pl = model(b, m, n=n)
            loss = policy_loss(pl, mv) + value_loss(vl, v)
            loss.backward()
            opt.step()
        # eval: move agreement + value agreement
        model.eval()
        correct = total = 0
        v_err = 0.0
        nv = 0
        with torch.no_grad():
            for b, m, mv, v in eval_ds:
                vl, pl = model(b, m, n=n)
                pred_m = pl.argmax(dim=1)
                correct += (pred_m == mv).sum().item()
                total += mv.size(0)
                sv = torch.sigmoid(vl.squeeze(-1))
                v_err += (sv - v).abs().sum().item()
                nv += v.size(0)
        acc = correct / max(total, 1)
        vmae = v_err / max(nv, 1)
        cfg = {**model.cfg(), "size_locked": False, "epoch": epoch}
        torch.save({"state_dict": model.state_dict(), "config": cfg}, args.save)
        log_progress(
            f"epoch {epoch}/{args.max_epochs} [{time.time()-t0:.0f}s] "
            f"eval move-agree={acc*100:.1f}% value-MAE={vmae:.3f} (saved)",
            args.progress,
        )

    cfg = {**model.cfg(), "size_locked": False}
    torch.save({"state_dict": model.state_dict(), "config": cfg}, args.save)
    log_progress(f"saved -> {args.save}", args.progress)

    # playout sanity: win rate vs random on the trained size
    model.eval()
    for sz in (3,):
        r = evaluate_vs_random(model, sz, games=40)
        log_progress(f"vs random ({sz}³): wins={r['wins']:.2f} losses={r['losses']:.2f} draws={r['draws']:.2f}", args.progress)


if __name__ == "__main__":
    main()