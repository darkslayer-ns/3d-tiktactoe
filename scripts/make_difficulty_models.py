#!/usr/bin/env python3
"""Create the 4 difficulty checkpoints from a strong base model.

Each difficulty shares the base weights; difficulty comes from the
wrong-move budget (how many random blunders per game) plus the per-difficulty
lookahead depth (already encoded in LOOKAHEAD_CONFIG in model_agent.py).
Checkpoints are saved in half precision to keep the files small.

Usage: python3 scripts/make_difficulty_models.py 4
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# difficulty -> (wrong-move budget per game, per-move blunder probability)
CONFIG = {
    "easy": (1, 0.5),
    "medium": (1, 0.5),
    "hard": (1, 0.5),
    "impossible": (0, 0.0),
}


def main():
    from training.torch_loader import load_model

    base_path = os.path.join(ROOT, f"model_{N}_base.pt")
    model, err = load_model(base_path, N)
    if err:
        print("base model error:", err)
        sys.exit(1)
    model.eval()
    for diff, (budget, p) in CONFIG.items():
        path = os.path.join(ROOT, f"model_{N}_{diff}.pt")
        torch.save(
            {
                "state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
                "config": {
                    "n": N,
                    "d_model": model.cfg()["d_model"],
                    "num_layers": model.cfg()["num_layers"],
                    "mistake_rate": p,
                    "shift_rate": 0.0,
                    "wrong_move_budget": budget,
                },
            },
            path,
        )
        print(f"saved {diff}: budget={budget} -> {path} ({os.path.getsize(path)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()