#!/usr/bin/env python3
"""Generate golden parity fixtures from PyTorch (CPU, float32).

For each cube size n in {3,4,5} and 3 deterministic boards, computes the
reference value + policy logits with the SAME weights file the C++ engine
loads. The C++ parity_test then replays these and must match to tolerance.

Run: python3 tools/make_fixtures.py [checkpoint] [output_dir]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.torch_loader import load_model, base_checkpoint_path


def gen_board(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random board with some empties; mask = empty cells, >=1 legal."""
    N = n**3
    p = rng.uniform(0.35, 0.6)  # density of occupied cells
    board = np.where(rng.random(N) < p, rng.integers(1, 3, N), 0).astype(np.int64)
    mask = board == 0
    if not mask.any():
        board[0] = 0
        mask[0] = True
    return board, mask


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else base_checkpoint_path(3)
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, err = load_model(ckpt, 3, device="cpu")
    if not model:
        print(f"load failed: {err}", file=sys.stderr)
        sys.exit(1)
    model.eval()

    rng = np.random.default_rng(20260830)
    for n in (3, 4, 5):
        for k in range(3):
            board, mask = gen_board(n, rng)
            bt = torch.tensor(board[None, :])
            mt = torch.tensor(mask[None, :])
            with torch.no_grad():
                v, pl = model(bt, mt, n=n)
            path = out_dir / f"fixture_{n}_{k}.bin"
            with open(path, "wb") as f:
                f.write(struct.pack("<ii", n, n**3))
                f.write(board.astype("<i4").tobytes())
                f.write(mask.astype("u1").tobytes())
                f.write(np.asarray(v.item(), dtype="<f4").tobytes())
                f.write(pl.squeeze(0).numpy().astype("<f4").tobytes())
            print(f"  {path.name}: n={n} value={v.item():+.4f} "
                  f"legal={mask.sum().item()}/{n**3}")


if __name__ == "__main__":
    main()