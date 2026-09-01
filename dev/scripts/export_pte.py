#!/usr/bin/env python3
"""Export the trained model to an ExecuTorch .pte file for on-device inference.

Usage: python3 scripts/export_pte.py
Output: mobile/assets/model.pte  (+ keeps the .onnx for reference/testing)

The exported program takes ONE input: the board as int32[1,27] (0/1/2), and
returns [value (sigmoid 0..1), policy_logits[1,27]]. The legal-move mask is
derived internally from the board (cells==0), so no bool input is needed —
ExecuTorch handles bool inputs poorly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from executorch import exir
from executorch.exir.backend.partitioner import Partitioner  # noqa
from executorch.exir.backend.backend_details import CompileSpec  # noqa


def build_wrapper():
    """A module that masks the policy head from the board inside the graph."""
    from training.torch_loader import load_model, base_checkpoint_path

    model, err = load_model(base_checkpoint_path(3), 3)
    if err:
        raise SystemExit(f"load failed: {err}")
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, board):  # board: int[1,27]
            # need the real mask for the inner model; build from board
            # model.forward expects (x, mask) and masks policy itself.
            mask = board == 0
            v, p = self.inner(board, mask)
            return v, p

    return Wrapper(model)


def main():
    wrapper = build_wrapper()
    example = torch.zeros(1, 27, dtype=torch.long)
    program = exir.to_edge(torch.export.export(wrapper, (example,))).to_executorch()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mobile", "assets")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.pte")
    with open(path, "wb") as f:
        f.write(program.buffer)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()