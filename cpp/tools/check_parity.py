#!/usr/bin/env python3
"""Randomized cross-check: C++ libmodel.so (ctypes) vs PyTorch CPU float32.

Covers every size the model can see (3..6) with 50 random boards each,
comparing value logits and finite policy logits. The golden fixture test in
tests/parity.cpp is the CI gate; this is the broader fuzz sweep.

Run: python3 tools/check_parity.py [checkpoint] [libmodel.so]
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.torch_loader import load_model, base_checkpoint_path


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else base_checkpoint_path(3)
    weights_bin = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
    lib_path = sys.argv[3] if len(sys.argv) > 3 else "build/libmodel.so"

    model, err = load_model(ckpt, 3, device="cpu")
    if not model:
        print(f"load failed: {err}", file=sys.stderr)
        sys.exit(1)
    model.eval()

    lib = ctypes.CDLL(str(Path(lib_path).resolve()))
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
    lib.tfm_free.argtypes = [ctypes.c_void_p]

    handle = ctypes.c_void_p()
    rc = lib.tfm_load(weights_bin.encode(), ctypes.byref(handle))
    if rc != 0:
        print(f"tfm_load failed rc={rc}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(7)
    max_v = max_p = 0.0
    n_checked = 0
    for n in (3, 4, 5, 6):
        N = n**3
        for _ in range(50):
            board = np.where(rng.random(N) < rng.uniform(0.3, 0.65),
                             rng.integers(1, 3, N), 0).astype(np.int64)
            mask = board == 0
            if not mask.any():
                board[0] = 0
                mask[0] = True

            bt = torch.tensor(board[None, :])
            mt = torch.tensor(mask[None, :])
            with torch.no_grad():
                v, pl = model(bt, mt, n=n)
            ref_v = float(v.item())
            ref_p = pl.squeeze(0).numpy().astype(np.float32)

            cb = (ctypes.c_int32 * N)(*board.tolist())
            cm = (ctypes.c_uint8 * N)(*mask.astype(np.uint8).tolist())
            cv = ctypes.c_float()
            cp = (ctypes.c_float * N)()
            rc = lib.tfm_forward(handle, cb, cm, n, ctypes.byref(cv), cp)
            assert rc == 0

            got_v = cv.value
            got_p = np.array(cp[:N], dtype=np.float32)

            max_v = max(max_v, abs(got_v - ref_v))
            finite = np.isfinite(ref_p)
            if finite.any():
                max_p = max(max_p, float(np.max(np.abs(got_p[finite] - ref_p[finite]))))
            # masked entries must be -inf in both
            if not np.all(np.isneginf(got_p[~finite])):
                print(f"  n={n}: masked policy not -inf in C++")
                sys.exit(1)
            n_checked += 1

    lib.tfm_free(handle)
    print(f"checked {n_checked} positions (sizes 3-6)")
    print(f"  max |d| value = {max_v:.3e}")
    print(f"  max |d| policy = {max_p:.3e}")
    ok = max_v < 1e-3 and max_p < 1e-3
    print("RANDOM PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()