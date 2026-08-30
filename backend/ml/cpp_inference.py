"""C++ transformer inference for the game server.

The server no longer runs PyTorch. Every forward pass goes through
`libmodel.so` — the modular C++ engine in cpp/ — loaded once per process and
called via ctypes. Parity with the trained weights is enforced by
`cpp/tools/parity.sh` (golden fixtures + randomized cross-check vs PyTorch).

    engine, err = load_cpp_model(n)          # (CppModel, None) | (None, error)
    value_logit, policy_logits = engine.eval_position(cells, player)

`eval_position` re-normalizes the board so the side-to-move is always token 1
(0 stays empty, player -> 1, opponent -> 2), then masks to empty cells — the
exact input convention the PyTorch model saw during training. All temporaries
live in the C++ frame, so concurrent calls are safe.
"""

from __future__ import annotations

import ctypes
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "cpp" / "model.bin"
DEFAULT_LIB = ROOT / "cpp" / "build" / "libmodel.so"

_lib = None
_handle = None
_loaded_weights = None


def _ensure_loaded(weights: Path) -> None:
    global _lib, _handle, _loaded_weights
    if _handle is not None and _loaded_weights == weights:
        return
    if not weights.exists():
        raise FileNotFoundError(
            f"no weights at {weights} — run cpp/tools/export_weights.py"
        )
    if not DEFAULT_LIB.exists():
        raise FileNotFoundError(f"no {DEFAULT_LIB} — build cpp/ with cmake")

    lib = ctypes.CDLL(str(DEFAULT_LIB))
    lib.tfm_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.tfm_load.restype = ctypes.c_int
    lib.tfm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tfm_forward.restype = ctypes.c_int
    lib.tfm_numel.argtypes = [ctypes.c_void_p]
    lib.tfm_numel.restype = ctypes.c_int
    lib.tfm_free.argtypes = [ctypes.c_void_p]

    handle = ctypes.c_void_p()
    rc = lib.tfm_load(str(weights).encode(), ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"tfm_load failed rc={rc}")
    _lib, _handle, _loaded_weights = lib, handle, weights


class CppModel:
    """Forward engine over the C++ transformer. Thread-safe."""

    def __init__(self, n: int, weights: Path = DEFAULT_WEIGHTS):
        _ensure_loaded(weights)
        self.n = n
        self.numel = _lib.tfm_numel(_handle)

    def eval_position(self, cells, player: int, n: int | None = None):
        """(value_logit, policy_logits: list[float]) for one position.

        `cells` holds RAW board values {0,1,2}; the side-to-move remap to
        tokens {0,1,2} happens here so both the C++ and torch engines receive
        byte-identical input (see cpp/tools/parity.sh).
        """
        size = n or self.n
        N = size**3
        cells = list(cells)
        assert len(cells) == N, f"expected {N} cells, got {len(cells)}"
        norm = [0 if c == 0 else (1 if c == player else 2) for c in cells]
        mask = [c == 0 for c in cells]

        cb = (ctypes.c_int32 * N)(*norm)
        cm = (ctypes.c_uint8 * N)(*mask)
        cv = ctypes.c_float()
        cp = (ctypes.c_float * N)()
        rc = _lib.tfm_forward(_handle, cb, cm, size, ctypes.byref(cv), cp)
        if rc != 0:
            raise RuntimeError(f"tfm_forward failed rc={rc}")
        return cv.value, [cp[i] for i in range(N)]


def load_cpp_model(board_n: int, weights=None):
    """(CppModel, None) or (None, error-string)."""
    try:
        return CppModel(board_n, Path(weights) if weights else DEFAULT_WEIGHTS), None
    except Exception as exc:
        return None, str(exc)


# ---- tiny numeric helpers (the server has no numpy/torch) ---------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(logits, temperature: float = 1.0) -> list:
    t = max(temperature, 1e-8)
    m = max(logits)
    exps = [math.exp((v - m) / t) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample_index(probs) -> int:
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def argmax(logits) -> int:
    best, best_i = logits[0], 0
    for i, v in enumerate(logits):
        if v > best:
            best, best_i = v, i
    return best_i