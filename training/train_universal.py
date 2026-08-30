"""Universal size-agnostic trainer.

Trains ONE ValuePolicyTransformer (coordinate-based position encoding) on
ANY mix of cube sizes, using policy-gradient self-play. The same weights
can then be applied to any board size at inference time.

Key point: the model takes `n` at forward time, so a batch can mix 3x3x3,
4x4x4, 6x6x6 samples. This lets a 6x6-trained model transfer to 3x3 (or be
fine-tuned there) — unlike a per-size model.

Usage:
    python3 -m training.train_universal --sizes 3,4,6 --games 400 --epochs 20
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from training.model import ValuePolicyTransformer, policy_loss, value_loss
from training.selfplay import pg_self_play_game
from training.eval import evaluate_vs_random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=str, default="3,4,6", help="comma-separated cube sizes")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1, help="self-play+train cycles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="mobile/assets/model_universal.pt")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ValuePolicyTransformer(n=sizes[0], d_model=args.d_model, num_layers=args.layers).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,} on {device}")
    print(f"sizes: {sizes}  games/size/cycle: {args.games}  cycles: {args.epochs}")

    for cycle in range(1, args.epochs + 1):
        t0 = time.time()
        # train ONE size at a time so batches are uniform (model is size-agnostic)
        for size in sizes:
            _train_size(model, size, args.games, args.batch, args.lr, args.temp, device)
        # eval: win rate vs random on each size
        rates = "  ".join(
            f"{s}³={evaluate_vs_random(model, s, games=30)['wins']:.2f}" for s in sizes
        )
        print(f"cycle {cycle}/{args.epochs} [{time.time()-t0:.0f}s] vs random: {rates}")

    torch.save({"state_dict": model.state_dict(), "config": model.cfg()}, args.save)
    print(f"saved -> {args.save}")


def _train_size(model, size, games, batch, lr, temp, device) -> float:
    """One self-play+train cycle for a single size (uniform batches)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buffer: list = []
    for _ in range(games):
        steps, winner = pg_self_play_game(model, size, temperature=temp)
        for x, m, idx, player in steps:
            adv = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            buffer.append(
                (x.to(device), m.to(device), idx, adv,
                 1.0 if adv > 0 else (0.5 if adv == 0 else 0.0))
            )
    random.shuffle(buffer)
    dev = next(model.parameters()).device
    losses = 0.0
    nbatch = 0
    for b in range(0, len(buffer) - (len(buffer) % batch), batch):
        sel = buffer[b : b + batch]
        xb = torch.stack([s[0] for s in sel])
        mb = torch.stack([s[1] for s in sel])
        mv = torch.tensor([s[2] for s in sel], device=dev)
        av = torch.tensor([s[3] for s in sel], dtype=torch.float32, device=dev)
        vt = torch.tensor([s[4] for s in sel], dtype=torch.float32, device=dev)
        opt.zero_grad()
        vl, pl = model(xb, mb, n=size)
        logp = torch.log_softmax(pl, dim=1).gather(1, mv.unsqueeze(1)).squeeze(1)
        loss = -(logp * av).mean() + value_loss(vl, vt)
        loss.backward()
        opt.step()
        losses += loss.item()
        nbatch += 1
    return losses / max(nbatch, 1)


if __name__ == "__main__":
    main()