"""Generate reference forward outputs from PyTorch for Dart verification."""
import json
import sys

sys.path.insert(0, ".")
from training.torch_loader import load_model, base_checkpoint_path
import torch


def state_to_inputs(model, cells):
    dev = next(model.parameters()).device
    x = torch.tensor([cells], dtype=torch.long, device=dev)
    mask = torch.tensor([[c == 0 for c in cells]], dtype=torch.bool, device=dev)
    with torch.no_grad():
        vlogits, plogits = model(x, mask)
    return (
        torch.sigmoid(vlogits).item(),
        torch.softmax(plogits[0], dim=0).tolist(),
    )


def main():
    model, err = load_model(base_checkpoint_path(3), 3)
    assert err is None, err
    model.eval()

    boards = [
        [0] * 27,
        [1] + [0] * 26,
        [1, 0, 0] + [2, 0, 0] + [0] * 22,
        [1] * 3 + [2, 0, 2, 1, 0, 0] + [0] * 18,
    ]
    out = []
    for cells in boards:
        v, p = state_to_inputs(model, cells)
        out.append({"cells": cells, "value": v, "policy": p})
    with open("scripts/dart_ref.json", "w") as f:
        json.dump(out, f)
    print("wrote scripts/dart_ref.json with", len(out), "cases")


if __name__ == "__main__":
    main()