#!/usr/bin/env python3
"""Convert mobile-rn/src/__tests__/fixtures/parity.json into the plain-text
format consumed by cpp/tests/search_parity.cpp.

The C++ search replay needs, per case:
  - the position + ai + the difficulty parameters the PYTHON backend used
    when the fixture was recorded (those determined the eval call sequence),
  - the captured eval_position call sequence — player, raw cells, and the
    (rounded) value/policy outputs, in order. The replay CONSUMES these
    outputs, so its traversal follows the exact path Python took (the
    generator already proved the rounded values are decision-identical).

The Python difficulty table is duplicated here deliberately: it is the
generation-time truth for the fixture, NOT the runtime config (the TS
DIFFICULTY table in mobile-rn/src/ai/mover.ts owns runtime behavior and has
evolved since these cases were recorded).

Run from the repo root:
    python3 cpp/tools/convert_parity_fixture.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# backend/ml/model_agent.py DIFFICULTY — generation-time values (frozen).
PY_DIFFICULTY = {
    "easy": {"depth": 2, "top_k": 3, "max_nodes": 220},
    "medium": {"depth": 3, "top_k": 3, "max_nodes": 220},
    "hard": {"depth": 4, "top_k": 3, "max_nodes": 220},
}


def main() -> int:
    src = os.path.join(ROOT, "mobile-rn", "src", "__tests__", "fixtures", "parity.json")
    dst = os.path.join(ROOT, "cpp", "fixtures", "parity_search.txt")
    with open(src) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(f"CASES {len(data['cases'])}\n")
        for c in data["cases"]:
            cfg = PY_DIFFICULTY[c["difficulty"]]
            f.write(f"CASE {c['size']} {c['difficulty']} {c['ai']}\n")
            f.write(f"PARAMS {cfg['depth']} {cfg['top_k']} {cfg['max_nodes']}\n")
            f.write("POS " + " ".join(str(x) for x in c["position"]) + "\n")
            f.write(f"CAPTURE {len(c['capture'])}\n")
            for e in c["capture"]:
                f.write(f"EVAL {e['player']} {e['value']!r}\n")
                f.write("BOARD " + " ".join(str(x) for x in e["cells"]) + "\n")
                f.write("POLICY " + " ".join(str(x) for x in e["policy"]) + "\n")
            f.write(f"EXPECTED {c['expectedMove']}\n")

    n_evals = sum(len(c["capture"]) for c in data["cases"])
    print(f"wrote {dst} ({len(data['cases'])} cases, {n_evals} captured evals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
