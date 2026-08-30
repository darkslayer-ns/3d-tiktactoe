"""Torch checkpoint loading + adapter — used ONLY by training/lab tooling.

The game server does NOT import this module: it runs the C++ engine
(backend/ml/cpp_inference.py) and must stay torch-free. Training,
calibration, verification and export scripts need the original PyTorch graph,
so the loaders and a torch adapter live here.

The `TorchModelAdapter` exposes the same `eval_position(cells, player, n)`
seam as the C++ `CppModel`, so movers/predictors work identically whether
they are fed a torch graph or the C++ engine.
"""

from __future__ import annotations

import os

import torch

from backend.game.board import EMPTY

# ONE universal checkpoint: the model is size-agnostic (coordinate-based
# position encoding), so a single file serves every cube size.
UNIVERSAL_CHECKPOINT = "model_universal.pt"

FALLBACK_ARCH = {"d_model": 64, "num_layers": 2}


def checkpoint_path(difficulty: str, n: int) -> str:
    # kept for compatibility; difficulty is applied at runtime (mistake rate)
    return UNIVERSAL_CHECKPOINT


def base_checkpoint_path(n: int) -> str:
    return UNIVERSAL_CHECKPOINT


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str, board_n: int, device: str | None = None):
    """Load a trained checkpoint. Returns (model, None) or (None, error)."""
    from training.model import ValuePolicyTransformer

    if not os.path.exists(path):
        return None, f"no checkpoint at {path} — train a model first"
    device = device or default_device()
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and "state_dict" in ck:
        cfg = ck.get("config", {})
        d_model = cfg.get("d_model", FALLBACK_ARCH["d_model"])
        layers = cfg.get("num_layers", FALLBACK_ARCH["num_layers"])
        n = cfg.get("n", board_n)
        mistake_rate = cfg.get("mistake_rate", 0.0)
        shift_rate = cfg.get("shift_rate", 0.0)
        wrong_move_budget = int(cfg.get("wrong_move_budget", 0))
    else:
        d_model, layers, n = FALLBACK_ARCH["d_model"], FALLBACK_ARCH["num_layers"], board_n
        mistake_rate = 0.0
        shift_rate = 0.0
        wrong_move_budget = 0
    if cfg.get("size_locked"):
        if n != board_n:
            return None, f"model trained for {n}×{n}×{n}, not {board_n}×{board_n}×{board_n}"
    model = ValuePolicyTransformer(board_n, d_model=d_model, num_layers=layers).to(device)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    # checkpoints may be saved in half precision to keep file size down
    state = {k: v.float() for k, v in state.items()}
    try:
        model.load_state_dict(state)
    except Exception as exc:
        return None, f"checkpoint load failed: {exc}"
    model.eval()
    model.mistake_rate = mistake_rate
    model.shift_rate = shift_rate
    model.wrong_move_budget = wrong_move_budget
    try:
        model.fingerprint = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        model.fingerprint = (0, 0)
    return model, None


class TorchModelAdapter:
    """Wrap a torch ValuePolicyTransformer with the eval_position() seam."""

    def __init__(self, torch_model):
        self._m = torch_model

    @property
    def n(self) -> int:
        return self._m.n

    def eval_position(self, cells, player: int, n: int | None = None):
        dev = next(self._m.parameters()).device
        norm = [c if c == EMPTY else (1 if c == player else 2) for c in cells]
        x = torch.tensor(norm, dtype=torch.long, device=dev).unsqueeze(0)
        m = torch.tensor([c == EMPTY for c in cells], dtype=torch.bool, device=dev).unsqueeze(0)
        self._m.eval()
        with torch.no_grad():
            value, policy = self._m(x, m, n=n)
        return value.item(), policy.squeeze(0).tolist()