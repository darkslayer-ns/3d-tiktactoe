#!/usr/bin/env python3
"""Generate the AI parity fixture from the real Python backend.

For each (size, position, difficulty) case: seeds Python's random, interposes
on the engine's eval_position to capture every forward pass, runs the real
LookaheadMover (C++ engine), and records the chosen move plus the exact eval
capture and the RNG draw log.

The TS parity test (src/__tests__/parity.test.ts) replays each case through
the ported mover with a mock engine that answers from the capture table and a
scripted RNG that pops the recorded draws — byte-for-byte game parity.

Run from the repo root:
    python3 mobile-rn/scripts/gen_parity_fixture.py
"""

import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mobile-rn
REPO = os.path.dirname(ROOT)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from backend.game.board import Board, P1  # noqa: E402
from backend.ml.cpp_inference import load_cpp_model  # noqa: E402
from backend.ml.model_agent import LookaheadMover  # noqa: E402
from backend.ml.predictor import OpponentPredictor  # noqa: E402

# --- RNG recorder ---------------------------------------------------------
# Replace random.random / random.choice with recorders that delegate to the
# real (seeded) PRNG but log every draw the algorithm makes. Because the log
# stores the OBSERVED values, TS replay never needs to reproduce the MT19937
# stream — it just pops the recorded floats / choice indices in the same order.
_orig_random = random.random
_orig_randrange = random.randrange
rng_log: list = []


def _rec_random():
    v = _orig_random()
    rng_log.append(["random", v])
    return v


def _rec_choice(seq):
    idx = _orig_randrange(len(seq))
    rng_log.append(["choice", len(seq), idx])
    return seq[idx]


random.random = _rec_random
random.choice = _rec_choice

# model_agent binds `import random as _random` — the module object is shared,
# so patched random.random/choice are what it calls. Keep the alias too.
import backend.ml.model_agent as _ma  # noqa: E402

_ma._random = random


# --- eval capture ---------------------------------------------------------
class CaptureEngine:
    """Wraps the real C++ model; records every eval_position call."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    @property
    def n(self):
        return self.inner.n

    def eval_position(self, cells, player, n=None):
        v, p = self.inner.eval_position(list(cells), player, n)
        self.calls.append(
            {
                "cells": [int(c) for c in cells],
                "player": int(player),
                "n": int(n or self.inner.n),
                "value": float(v),
                "policy": [float(x) for x in p],
            }
        )
        return v, p


class RoundedEngine:
    """Answers eval_position straight from a stored (rounded) capture, so the
    verification re-run provably consumes the exact values committed to JSON.

    Non-finite policy logits (occupied-cell masks) are stored as ±1e9 —
    behaviorally identical to ±inf for softmax/argmax/sigmoid (exp(-1e9)=0,
    never the max), but finite so the JSON parses in JS.
    """

    def __init__(self, capture):
        self.capture = capture
        self.table = {}
        for c in capture:
            key = (tuple(c["cells"]), c["player"], c["n"])
            self.table.setdefault(key, (c["value"], c["policy"]))

    @property
    def n(self):
        return self.capture[0]["n"]

    def eval_position(self, cells, player, n=None):
        v, p = self.table[(tuple(cells), player, n or self.n)]
        return float(v), [float(x) for x in p]


# --- fixture positions ----------------------------------------------------
def empty(n):
    return [0] * (n**3)


def mid3_1():
    cells = [0] * 27
    for idx, pl in [(0, 1), (1, 2), (4, 1), (13, 2), (9, 1), (10, 2), (26, 1), (14, 2)]:
        cells[idx] = pl
    return cells


def mid3_2():
    cells = [0] * 27
    for idx, pl in [(0, 1), (2, 2), (13, 1), (12, 2), (20, 1), (8, 2), (5, 1), (16, 2), (22, 1)]:
        cells[idx] = pl
    return cells


def mid4_1():
    cells = [0] * 64
    for idx, pl in [(0, 1), (5, 2), (21, 1), (42, 2), (9, 1), (14, 2), (63, 1), (32, 2), (8, 1), (37, 2)]:
        cells[idx] = pl
    return cells


# --- case runner ----------------------------------------------------------
def _finite(x, digits):
    r = round(x, digits)
    if not math.isfinite(r):
        return 1e9 if r > 0 else -1e9
    return r


def run_case(size, cells, difficulty, ai, seed, round_digits=4):
    rng_log.clear()
    random.seed(seed)
    board = Board(size, cells)
    over = board.outcome()[2]
    assert not over, "fixture position must not already be terminal"
    model, err = load_cpp_model(size)
    if err:
        raise RuntimeError(f"no model for n={size}: {err}")
    engine = CaptureEngine(model)
    pred = OpponentPredictor(board, engine)
    mover = LookaheadMover(engine, board, pred, difficulty=difficulty)
    chosen = mover(ai)
    rng_full = [list(x) for x in rng_log]

    # Round the captured floats so the committed JSON stays small, then PROVE
    # the rounded values still make the mover pick the same move with the same
    # RNG draw sequence (a decision-identical replay of the full precision).
    if round_digits is None:
        capture = engine.calls
    else:
        capture = [
            {
                "cells": c["cells"],
                "player": c["player"],
                "n": c["n"],
                "value": _finite(c["value"], round_digits),
                "policy": [_finite(x, round_digits) for x in c["policy"]],
            }
            for c in engine.calls
        ]
        rng_log.clear()
        random.seed(seed)
        board2 = Board(size, cells)
        rounded = RoundedEngine(capture)
        pred2 = OpponentPredictor(board2, rounded)
        mover2 = LookaheadMover(rounded, board2, pred2, difficulty=difficulty)
        chosen2 = mover2(ai)
        rng_rounded = [list(x) for x in rng_log]
        assert chosen2 == chosen, f"rounding changed move for n={size}/{difficulty}"
        assert rng_rounded == rng_full, f"rounding changed rng path for n={size}/{difficulty}"

    return {
        "size": size,
        "position": [int(c) for c in board.cells],
        "ai": int(ai),
        "difficulty": difficulty,
        "capture": capture,
        "rng": rng_full,
        "expectedMove": int(chosen),
    }


def main():
    seed = 42
    cases = [
        run_case(3, empty(3), "easy", P1, seed),
        run_case(3, empty(3), "hard", P1, seed + 1),
        run_case(3, mid3_1(), "easy", P1, seed + 2),
        run_case(3, mid3_2(), "hard", P1, seed + 3),
        run_case(4, mid4_1(), "easy", P1, seed + 4),
    ]

    out = {"generator": "backend/ml parity fixture (python)", "cases": cases}
    path = os.path.join(ROOT, "src", "__tests__", "fixtures", "parity.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)

    total_evals = sum(len(c["capture"]) for c in cases)
    print(f"wrote {path}")
    print(f"cases: {len(cases)}, total eval calls: {total_evals}, file size: {os.path.getsize(path)} bytes")
    for c in cases:
        empties = sum(1 for x in c["position"] if x == 0)
        print(
            f"  n={c['size']} {c['difficulty']:6s} ai={c['ai']} "
            f"empties={empties} -> move {c['expectedMove']} "
            f"(rng draws: {len(c['rng'])}, evals: {len(c['capture'])})"
        )


if __name__ == "__main__":
    main()